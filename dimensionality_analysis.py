import matplotlib.pyplot as plt
import numpy as np
import torch

# pip install umap-learn ripser
from datasets import load_dataset
from persim import plot_diagrams
from ripser import ripser
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE

# ---- Config ----
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
n_batches = 100
hook_point = "blocks.8.hook_resid_post"

# ---- Load model and SAE ----
model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m", device=device)
sae, cfg_dict, sparsity = SAE.load_from_disk(
    path="/scratch/Collin/SAELens/checkpoints/5f0ldf78/final_250003456",
    device=device,
)

n_experts = sae.cfg.n_experts
d_expert = sae.cfg.d_expert
k_experts = sae.cfg.k_experts

# ---- Load dataset ----
dataset = load_dataset("openwebtext", split="train", streaming=True)
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

# ---- Collect activations ----
expert_latents = {i: [] for i in range(n_experts)}
expert_counts = torch.zeros(n_experts)

iterator = iter(dataset)
for batch_idx in tqdm(range(n_batches), desc="Collecting activations"):
    texts = [next(iterator)["text"] for _ in range(batch_size)]
    tokens = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=128
    )["input_ids"].to(device)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
        residual = cache[hook_point]

    residual_flat = residual.reshape(-1, residual.shape[-1])

    with torch.no_grad():
        feature_acts = sae.encode(residual_flat)
        feature_acts_by_expert = feature_acts.view(-1, n_experts, d_expert)

    active_mask = feature_acts_by_expert.abs().sum(dim=-1) > 0

    for expert_id in range(n_experts):
        mask = active_mask[:, expert_id]
        if mask.any():
            latents = feature_acts_by_expert[mask, expert_id].cpu().numpy()
            expert_latents[expert_id].extend(latents)
            expert_counts[expert_id] += mask.sum().item()


# ---- Compute dimensionality for all experts ----
def analyze_expert_dimensionality(expert_id, latent_list, min_samples=50):
    if len(latent_list) < min_samples:
        return None
    latents = np.stack(latent_list)
    latents_centered = latents - latents.mean(axis=0)
    U, S, Vt = np.linalg.svd(latents_centered, full_matrices=False)
    participation_ratio = (S.sum() ** 2) / (S**2).sum()
    variance_explained = S**2 / (S**2).sum()
    cumulative_variance = np.cumsum(variance_explained)
    dim_90 = int(np.searchsorted(cumulative_variance, 0.90) + 1)
    dim_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    active_dims = int((variance_explained > 0.01).sum())
    return {
        "expert_id": expert_id,
        "n_samples": len(latent_list),
        "participation_ratio": float(participation_ratio),
        "dim_90_variance": dim_90,
        "dim_95_variance": dim_95,
        "active_dims": active_dims,
        "singular_values": S,
        "variance_explained": variance_explained,
        "cumulative_variance": cumulative_variance,
    }


results = []
for expert_id in tqdm(range(n_experts), desc="Analyzing experts"):
    result = analyze_expert_dimensionality(expert_id, expert_latents[expert_id])
    if result is not None:
        results.append(result)

sorted_by_pr = sorted(results, key=lambda r: r["participation_ratio"])

# ---- Select interesting experts ----
# Low-dim but not trivially 1D, with enough samples
interesting = [
    r
    for r in sorted_by_pr
    if 2.5 < r["participation_ratio"] < 9.0 and r["n_samples"] >= 200
]

# Also include the near-1D ones for comparison
near_1d = [
    r for r in sorted_by_pr if r["participation_ratio"] < 2.5 and r["n_samples"] >= 200
]

# Pick targets
targets = []
if near_1d:
    targets.append(near_1d[0])
targets.extend(interesting[:5])  # up to 5 interesting experts

print(f"\nSelected {len(targets)} experts for manifold analysis:")
for r in targets:
    print(
        f"  Expert {r['expert_id']}: PR={r['participation_ratio']:.2f}, "
        f"samples={r['n_samples']}"
    )


# ---- Manifold Analysis Functions ----
def prepare_data(expert_id, max_points=5000, max_points_tda=1000):
    """Prepare and subsample data for an expert."""
    latents = np.stack(expert_latents[expert_id])
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    # Subsample for visualization
    if latents_scaled.shape[0] > max_points:
        vis_idx = np.random.choice(latents_scaled.shape[0], max_points, replace=False)
        latents_vis = latents_scaled[vis_idx]
    else:
        latents_vis = latents_scaled

    # Smaller subsample for TDA (persistent homology is expensive)
    if latents_scaled.shape[0] > max_points_tda:
        tda_idx = np.random.choice(
            latents_scaled.shape[0], max_points_tda, replace=False
        )
        latents_tda = latents_scaled[tda_idx]
    else:
        latents_tda = latents_scaled

    return latents_scaled, latents_vis, latents_tda


# def run_umap(data, n_neighbors=15, min_dist=0.1):
#     reducer = umap.UMAP(
#         n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42
#     )
#     return reducer.fit_transform(data)


def run_tsne(data, perplexity=30):
    reducer = TSNE(
        n_components=2, perplexity=min(perplexity, data.shape[0] - 1), random_state=42
    )
    return reducer.fit_transform(data)


def run_tda(data, maxdim=2):
    """Run persistent homology up to dimension maxdim."""
    result = ripser(data, maxdim=maxdim)
    return result


# ---- Run analysis on each target expert ----
expert_analyses = {}
for r in tqdm(targets, desc="Running manifold analysis"):
    expert_id = r["expert_id"]
    print(f"\nAnalyzing expert {expert_id}...")

    latents_full, latents_vis, latents_tda = prepare_data(expert_id)

    analysis = {"info": r}

    # # UMAP
    # print("  Running UMAP...")
    # analysis["umap"] = run_umap(latents_vis)

    # t-SNE
    print("  Running t-SNE...")
    analysis["tsne"] = run_tsne(latents_vis)

    # TDA
    print("  Running persistent homology...")
    analysis["tda"] = run_tda(latents_tda, maxdim=2)

    # PCA for reference
    latents_centered = latents_vis - latents_vis.mean(axis=0)
    U, S, Vt = np.linalg.svd(latents_centered, full_matrices=False)
    analysis["pca_2d"] = latents_centered @ Vt[:2].T
    analysis["pca_3d"] = latents_centered @ Vt[:3].T

    expert_analyses[expert_id] = analysis

# ---- Plot: UMAP + t-SNE + PCA for each expert ----
n_targets = len(targets)
fig, axes = plt.subplots(n_targets, 3, figsize=(18, 5 * n_targets))
if n_targets == 1:
    axes = axes.reshape(1, -1)

for row, r in enumerate(targets):
    expert_id = r["expert_id"]
    analysis = expert_analyses[expert_id]

    # PCA
    pca_2d = analysis["pca_2d"]
    axes[row, 0].scatter(pca_2d[:, 0], pca_2d[:, 1], s=1, alpha=0.3, c="steelblue")
    axes[row, 0].set_title(
        f"Expert {expert_id} — PCA\nPR={r['participation_ratio']:.2f}"
    )
    axes[row, 0].set_xlabel("PC1")
    axes[row, 0].set_ylabel("PC2")

    # # UMAP
    # umap_2d = analysis["umap"]
    # axes[row, 1].scatter(umap_2d[:, 0], umap_2d[:, 1], s=1, alpha=0.3, c="steelblue")
    # axes[row, 1].set_title(f"Expert {expert_id} — UMAP")
    # axes[row, 1].set_xlabel("UMAP1")
    # axes[row, 1].set_ylabel("UMAP2")

    # t-SNE
    tsne_2d = analysis["tsne"]
    axes[row, 2].scatter(tsne_2d[:, 0], tsne_2d[:, 1], s=1, alpha=0.3, c="steelblue")
    axes[row, 2].set_title(f"Expert {expert_id} — t-SNE")
    axes[row, 2].set_xlabel("t-SNE1")
    axes[row, 2].set_ylabel("t-SNE2")

plt.tight_layout()
plt.savefig("expert_manifold_embeddings.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved expert_manifold_embeddings.png")

# ---- Plot: Persistence diagrams ----
fig2, axes2 = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
if n_targets == 1:
    axes2 = [axes2]

for col, r in enumerate(targets):
    expert_id = r["expert_id"]
    tda_result = expert_analyses[expert_id]["tda"]
    plot_diagrams(tda_result["dgms"], ax=axes2[col], show=False)
    axes2[col].set_title(f"Expert {expert_id}\nPR={r['participation_ratio']:.2f}")

plt.tight_layout()
plt.savefig("expert_persistence_diagrams.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved expert_persistence_diagrams.png")

# ---- Print TDA summary ----
print("\n" + "=" * 60)
print("Topological Summary per Expert")
print("=" * 60)

for r in targets:
    expert_id = r["expert_id"]
    tda_result = expert_analyses[expert_id]["tda"]
    dgms = tda_result["dgms"]

    print(f"\nExpert {expert_id} (PR={r['participation_ratio']:.2f}):")

    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            print(f"  H{dim}: no features")
            continue

        # Filter out infinite death features for stats
        finite = dgm[dgm[:, 1] != np.inf]
        if len(finite) == 0:
            print(f"  H{dim}: {len(dgm)} features (all infinite)")
            continue

        lifetimes = finite[:, 1] - finite[:, 0]
        # Significant features: lifetime > mean + 1*std
        threshold = lifetimes.mean() + lifetimes.std()
        significant = lifetimes > threshold
        n_significant = significant.sum()

        print(
            f"  H{dim}: {len(dgm)} total features, {n_significant} significant "
            f"(lifetime > {threshold:.3f})"
        )
        if n_significant > 0:
            sig_lifetimes = lifetimes[significant]
            print(
                f"       Significant lifetimes: {sorted(sig_lifetimes, reverse=True)[:5]}"
            )

    # Interpretation hints
    h0_count = len(dgms[0]) if len(dgms) > 0 else 0
    h1_sig = 0
    h2_sig = 0
    if len(dgms) > 1:
        finite_h1 = dgms[1][dgms[1][:, 1] != np.inf]
        if len(finite_h1) > 0:
            lt = finite_h1[:, 1] - finite_h1[:, 0]
            h1_sig = (lt > lt.mean() + lt.std()).sum()
    if len(dgms) > 2:
        finite_h2 = dgms[2][dgms[2][:, 1] != np.inf]
        if len(finite_h2) > 0:
            lt = finite_h2[:, 1] - finite_h2[:, 0]
            h2_sig = (lt > lt.mean() + lt.std()).sum()

    hints = []
    if h1_sig > 0:
        hints.append(f"circular/loop structure (H1={h1_sig})")
    if h2_sig > 0:
        hints.append(f"spherical/void structure (H2={h2_sig})")
    if hints:
        print(f"  → Possible: {', '.join(hints)}")
