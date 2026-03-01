import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import umap
from datasets import load_dataset
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE

# ---- Config ----
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 512
n_batches = 100
hook_point = "blocks.8.hook_resid_post"
max_vis_points = 5000

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

# ---- Collect activations AND tokens ----
expert_latents = {i: [] for i in range(n_experts)}
expert_contexts = {i: [] for i in range(n_experts)}
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

    # Find active experts per token: (n_tokens, k_experts) indices
    active_norms = feature_acts_by_expert.norm(dim=-1)  # (n_tokens, n_experts)
    active_mask = active_norms > 0

    # Precompute all token strings for this batch
    batch_size_actual, seq_len = tokens.shape
    tokens_cpu = tokens.cpu()

    # Build context strings in bulk
    all_contexts_batch = []
    for b in range(batch_size_actual):
        for p in range(seq_len):
            start = max(0, p - 10)
            end = min(seq_len, p + 11)
            pre = tokenizer.decode(tokens_cpu[b, start:p].tolist())
            target = tokenizer.decode([tokens_cpu[b, p].item()])
            post = tokenizer.decode(tokens_cpu[b, p + 1 : end].tolist())
            all_contexts_batch.append(f"{pre}[{target}]{post}")

    # Batch collect per expert
    feature_acts_cpu = feature_acts_by_expert.cpu().numpy()
    active_mask_cpu = active_mask.cpu()

    for expert_id in active_mask_cpu.any(dim=0).nonzero(as_tuple=True)[0].tolist():
        mask = active_mask_cpu[:, expert_id]
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        expert_latents[expert_id].extend(feature_acts_cpu[indices, expert_id])
        expert_counts[expert_id] += len(indices)
        for idx in indices:
            expert_contexts[expert_id].append(all_contexts_batch[idx])

print("\nExpert utilization stats:")
print(f"  Active experts: {(expert_counts > 0).sum().item()}")
print(f"  Mean tokens per expert: {expert_counts.mean().item():.1f}")
print(f"  Min tokens: {expert_counts.min().item():.0f}")
print(f"  Max tokens: {expert_counts.max().item():.0f}")


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
interesting = [
    r
    for r in sorted_by_pr
    if 2.5 < r["participation_ratio"] < 9.0 and r["n_samples"] >= 200
]

near_1d = [
    r for r in sorted_by_pr if r["participation_ratio"] < 2.5 and r["n_samples"] >= 200
]

targets = []
if near_1d:
    targets.append(near_1d[0])
targets.extend(interesting[:5])

print(f"\nSelected {len(targets)} experts for manifold analysis:")
for r in targets:
    print(
        f"  Expert {r['expert_id']}: PR={r['participation_ratio']:.2f}, "
        f"samples={r['n_samples']}"
    )


# ---- Analysis Functions ----
def prepare_data(expert_id):
    latents = np.stack(expert_latents[expert_id])
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    n_total = latents_scaled.shape[0]

    if n_total > max_vis_points:
        vis_idx = np.random.choice(n_total, max_vis_points, replace=False)
    else:
        vis_idx = np.arange(n_total)

    latents_vis = latents_scaled[vis_idx]

    return latents_scaled, latents_vis, vis_idx


def run_umap(data, n_components=3, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_jobs=-1,
    )
    return reducer.fit_transform(data)


def run_tsne(data, n_components=3, perplexity=30):
    reducer = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, data.shape[0] - 1),
        n_jobs=-1,
    )
    return reducer.fit_transform(data)


def make_snippet(context, max_len=80):
    if len(context) <= max_len:
        return context
    bracket_pos = context.find("[")
    if bracket_pos >= 0:
        start = max(0, bracket_pos - max_len // 2)
        end = min(len(context), bracket_pos + max_len // 2)
        snippet = context[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(context):
            snippet = snippet + "..."
        return snippet
    return context[:max_len] + "..."


# ---- Run analysis on each target expert ----
expert_analyses = {}
for r in tqdm(targets, desc="Running manifold analysis"):
    expert_id = r["expert_id"]
    print(f"\nAnalyzing expert {expert_id}...")

    latents_full, latents_vis, vis_idx = prepare_data(expert_id)

    analysis = {"info": r, "vis_idx": vis_idx}

    print("  Running UMAP (3D)...")
    analysis["umap_3d"] = run_umap(latents_vis, n_components=3)

    print("  Running t-SNE (3D)...")
    analysis["tsne_3d"] = run_tsne(latents_vis, n_components=3)

    latents_centered = latents_vis - latents_vis.mean(axis=0)
    U, S, Vt = np.linalg.svd(latents_centered, full_matrices=False)
    analysis["pca_3d"] = latents_centered @ Vt[:3].T

    expert_analyses[expert_id] = analysis

# ---- Interactive 3D plots with context snippets on hover ----
for r in targets:
    expert_id = r["expert_id"]
    analysis = expert_analyses[expert_id]
    pr = r["participation_ratio"]
    vis_idx = analysis["vis_idx"]

    all_contexts_list = expert_contexts[expert_id]

    vis_snippets = []
    for i in vis_idx:
        if i < len(all_contexts_list):
            vis_snippets.append(make_snippet(all_contexts_list[i]))
        else:
            vis_snippets.append("(no context)")

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"PCA (PR={pr:.2f})",
            f"UMAP (PR={pr:.2f})",
            f"t-SNE (PR={pr:.2f})",
        ],
    )

    pca = analysis["pca_3d"]
    distances_pca = np.linalg.norm(pca, axis=1)
    fig.add_trace(
        go.Scatter3d(
            x=pca[:, 0],
            y=pca[:, 1],
            z=pca[:, 2],
            mode="markers",
            marker=dict(
                size=1.5, color=distances_pca, colorscale="Viridis", opacity=0.5
            ),
            text=vis_snippets,
            hoverinfo="text",
            name="PCA",
        ),
        row=1,
        col=1,
    )

    umap_3d = analysis["umap_3d"]
    distances_umap = np.linalg.norm(umap_3d - umap_3d.mean(axis=0), axis=1)
    fig.add_trace(
        go.Scatter3d(
            x=umap_3d[:, 0],
            y=umap_3d[:, 1],
            z=umap_3d[:, 2],
            mode="markers",
            marker=dict(
                size=1.5, color=distances_umap, colorscale="Viridis", opacity=0.5
            ),
            text=vis_snippets,
            hoverinfo="text",
            name="UMAP",
        ),
        row=1,
        col=2,
    )

    tsne_3d = analysis["tsne_3d"]
    distances_tsne = np.linalg.norm(tsne_3d - tsne_3d.mean(axis=0), axis=1)
    fig.add_trace(
        go.Scatter3d(
            x=tsne_3d[:, 0],
            y=tsne_3d[:, 1],
            z=tsne_3d[:, 2],
            mode="markers",
            marker=dict(
                size=1.5, color=distances_tsne, colorscale="Viridis", opacity=0.5
            ),
            text=vis_snippets,
            hoverinfo="text",
            name="t-SNE",
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=f"Expert {expert_id} — 3D Manifold Visualizations (PR={pr:.2f}, n={r['n_samples']})",
        height=700,
        width=1800,
        showlegend=False,
    )

    filename = f"expert_{expert_id}_3d.html"
    fig.write_html(filename)
    print(f"Saved interactive plot: {filename}")

# ---- Token analysis along manifold coordinates ----
print("\n" + "=" * 60)
print("Token Analysis Along Manifold Coordinates")
print("=" * 60)

for r in targets:
    expert_id = r["expert_id"]
    all_ctxs = expert_contexts[expert_id]

    if len(all_ctxs) < 50:
        print(f"\nExpert {expert_id}: too few tokens for analysis")
        continue

    latents = np.stack(expert_latents[expert_id])
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)
    latents_centered = latents_scaled - latents_scaled.mean(axis=0)
    U, S, Vt = np.linalg.svd(latents_centered, full_matrices=False)
    pc1 = latents_centered @ Vt[0]

    sort_idx = np.argsort(pc1)
    n_total = len(sort_idx)
    n_bins = 10
    bin_size = n_total // n_bins

    print(f"\nExpert {expert_id} (PR={r['participation_ratio']:.2f})")
    print(f"Tokens sorted by PC1 coordinate, showing {n_bins} bins:")
    print("-" * 80)

    for bin_i in range(n_bins):
        start = bin_i * bin_size
        end = start + bin_size
        bin_indices = sort_idx[start:end]

        pc1_min = pc1[bin_indices].min()
        pc1_max = pc1[bin_indices].max()

        sample_size = min(5, len(bin_indices))
        sample_indices = np.random.choice(bin_indices, sample_size, replace=False)

        print(f"\n  Bin {bin_i} (PC1: {pc1_min:.2f} to {pc1_max:.2f}):")
        for idx in sample_indices:
            if idx < len(all_ctxs):
                print(f"    {make_snippet(all_ctxs[idx], max_len=100)}")

    print("\n  --- Extreme LOW PC1 tokens ---")
    for idx in sort_idx[:10]:
        if idx < len(all_ctxs):
            print(
                f"    PC1={pc1[idx]:>7.2f}  {make_snippet(all_ctxs[idx], max_len=100)}"
            )

    print("\n  --- Extreme HIGH PC1 tokens ---")
    for idx in sort_idx[-10:]:
        if idx < len(all_ctxs):
            print(
                f"    PC1={pc1[idx]:>7.2f}  {make_snippet(all_ctxs[idx], max_len=100)}"
            )

# ---- Dimensionality summary plots ----
participation_ratios = [r["participation_ratio"] for r in results]
dims_90 = [r["dim_90_variance"] for r in results]

fig_summary, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(participation_ratios, bins=30, edgecolor="black")
axes[0].set_xlabel("Participation Ratio")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Intrinsic Dimensionality")
axes[0].axvline(
    x=np.median(participation_ratios),
    color="r",
    linestyle="--",
    label=f"Median: {np.median(participation_ratios):.2f}",
)
axes[0].legend()

axes[1].hist(dims_90, bins=range(1, d_expert + 2), edgecolor="black", align="left")
axes[1].set_xlabel("Dimensions")
axes[1].set_ylabel("Count")
axes[1].set_title("Dimensions Needed for 90% Variance")

plt.tight_layout()
plt.savefig("expert_dimensionality_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved expert_dimensionality_summary.png")
