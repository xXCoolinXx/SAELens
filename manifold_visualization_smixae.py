import numpy as np
import plotly.graph_objects as go
import torch
from datasets import load_dataset
from plotly.subplots import make_subplots
from transformer_lens import HookedTransformer

from sae_lens import SAE

# ============================================================
# Config
# ============================================================
MODEL_NAME = "gemma-2-2b"
HOOK_NAME = "blocks.12.hook_resid_post"
CHECKPOINT_PATH = "/scratch/Collin/SAELens/checkpoints/wq3gukf1/62500864"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TEXTS = 1000
SEQ_LEN = 128
N_EXPERTS_TO_PLOT = 50
CONTEXT_WINDOW = 10  # tokens before/after for context


# ============================================================
# 1. Collect activations + token info
# ============================================================
print("Loading model...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)

print("Loading dataset...")
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

all_activations = []
all_token_ids = []
all_positions = []
all_text_indices = []

with torch.no_grad():
    for i, example in enumerate(dataset):
        if i >= N_TEXTS:
            break
        tokens = model.to_tokens(example["text"], prepend_bos=True)[:, :SEQ_LEN]
        _, cache = model.run_with_cache(tokens, names_filter=HOOK_NAME)
        acts = cache[HOOK_NAME].squeeze(0)  # (seq, d_model)

        seq_len = acts.shape[0]
        all_activations.append(acts.cpu())
        all_token_ids.append(tokens.squeeze(0).cpu())
        all_positions.extend(range(seq_len))
        all_text_indices.extend([i] * seq_len)

activations = torch.cat(all_activations, dim=0).to(DEVICE)
token_ids = torch.cat(all_token_ids, dim=0)  # (total_tokens,)
positions = np.array(all_positions)
text_indices = np.array(all_text_indices)

# Build per-text token lists for context lookup
text_tokens = {}
offset = 0
for i, toks in enumerate(all_token_ids):
    text_tokens[i] = toks
    offset += len(toks)

print(f"Collected {activations.shape[0]} activation vectors")


# ============================================================
# 2. Build hover labels: [token] with context
# ============================================================


def make_hover_label(text_idx, pos, token_ids_list, tokenizer, context_window=5):
    """Build '[token] ...context...' string for hover."""
    toks = text_tokens[text_idx]
    seq_len = len(toks)

    # The target token
    target_str = tokenizer.decode([toks[pos].item()])

    # Context before
    start = max(0, pos - context_window)
    end = min(seq_len, pos + context_window + 1)

    context_ids = toks[start:end].tolist()
    context_strs = [tokenizer.decode([t]) for t in context_ids]

    # Mark the target token with brackets
    target_offset = pos - start
    context_strs[target_offset] = f"[{context_strs[target_offset]}]"

    context = "".join(context_strs)
    # Clean for HTML hover
    context = context.replace("<", "&lt;").replace(">", "&gt;")
    context = context.replace("\n", "\\n")

    return context


print("Building hover labels...")
hover_labels = []
for i in range(len(positions)):
    label = make_hover_label(
        text_indices[i], positions[i], token_ids, model.tokenizer, CONTEXT_WINDOW
    )
    hover_labels.append(label)

hover_labels = np.array(hover_labels)


# ============================================================
# 3. Load SAE
# ============================================================
print("Loading SAE...")
sae = SAE.load_from_disk(
    path=CHECKPOINT_PATH,
    device=DEVICE,
)
# sae.eval()
print(f"Loaded SAE: {sae.cfg.n_experts} experts, d_bottleneck={sae.cfg.d_bottleneck}")


# ============================================================
# 4. Run activations through SAE
# ============================================================
print("Encoding activations...")
with torch.no_grad():
    z = sae.encode(activations)  # (n_tokens, n_experts, d_bottleneck)

z_np = z.cpu().numpy()
expert_norms = np.linalg.norm(z_np, axis=-1)  # (n_tokens, n_experts)

active_per_input = (expert_norms > 1e-6).sum(axis=1)
mean_norm_per_expert = expert_norms.mean(axis=0)
activation_freq = (expert_norms > 1e-6).mean(axis=0)

print(f"Mean active experts per input: {active_per_input.mean():.1f}")
print(
    f"Expert activation frequency: min={activation_freq.min():.3f}, "
    f"max={activation_freq.max():.3f}, mean={activation_freq.mean():.3f}"
)


# ============================================================
# 5. Select experts to plot
# ============================================================
top_by_freq = np.argsort(activation_freq)[::-1][:N_EXPERTS_TO_PLOT]

active_experts = np.where(activation_freq > 0.01)[0]
if len(active_experts) >= N_EXPERTS_TO_PLOT:
    random_experts = np.random.choice(active_experts, N_EXPERTS_TO_PLOT, replace=False)
else:
    random_experts = active_experts

expert_sets = {
    "top_by_frequency": top_by_freq,
    "random_active": random_experts,
}


# ============================================================
# 6. Plot
# ============================================================


def plot_experts(expert_ids, z_np, expert_norms, hover_labels, title, filename):
    n = len(expert_ids)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"}] * cols for _ in range(rows)],
        subplot_titles=[
            f"Expert {eid} (freq: {activation_freq[eid]:.2%}, "
            f"mean norm: {mean_norm_per_expert[eid]:.3f})"
            for eid in expert_ids
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for idx, expert_id in enumerate(expert_ids):
        row = idx // cols + 1
        col = idx % cols + 1

        norms = expert_norms[:, expert_id]
        active = norms > 1e-6
        n_active = active.sum()

        if n_active < 10:
            continue

        z_e = z_np[active, expert_id, :]  # (n_active, 3)
        norm_e = norms[active]
        labels_e = hover_labels[active]

        # Subsample if too many points
        max_points = 5000
        if n_active > max_points:
            subsample = np.random.choice(n_active, max_points, replace=False)
            z_e = z_e[subsample]
            norm_e = norm_e[subsample]
            labels_e = labels_e[subsample]

        fig.add_trace(
            go.Scatter3d(
                x=z_e[:, 0],
                y=z_e[:, 1],
                z=z_e[:, 2],
                mode="markers",
                marker=dict(
                    size=1.5,
                    opacity=0.6,
                    color=norm_e,
                    colorscale="Viridis",
                    showscale=(idx == 0),
                    colorbar=dict(title="||z||", x=1.02) if idx == 0 else None,
                ),
                name=f"Expert {expert_id}",
                customdata=labels_e,
                hovertemplate=(
                    f"<b>Expert {expert_id}</b><br>"
                    "z₁: %{x:.3f}<br>"
                    "z₂: %{y:.3f}<br>"
                    "z₃: %{z:.3f}<br>"
                    "norm: %{marker.color:.3f}<br>"
                    "<b>%{customdata}</b>"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        scene_name = f"scene{idx + 1}" if idx > 0 else "scene"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis_title="z₁",
                    yaxis_title="z₂",
                    zaxis_title="z₃",
                )
            }
        )

    fig.update_layout(
        height=400 * rows,
        width=400 * cols,
        title_text=title,
        showlegend=False,
    )

    fig.write_html(filename)
    print(f"Saved: {filename}")


for set_name, expert_ids in expert_sets.items():
    plot_experts(
        expert_ids=expert_ids,
        z_np=z_np,
        expert_norms=expert_norms,
        hover_labels=hover_labels,
        title=f"SMIXAE Expert Bottleneck Structures ({set_name})",
        filename=f"expert_manifolds_{set_name}.html",
    )


# ============================================================
# 7. Norm distribution
# ============================================================
fig_dist = go.Figure()

all_norms = expert_norms.flatten()
all_norms_nonzero = all_norms[all_norms > 1e-10]
log_norms = np.log10(all_norms_nonzero)

fig_dist.add_trace(
    go.Histogram(
        x=log_norms,
        nbinsx=100,
        name="All experts",
    )
)

fig_dist.update_layout(
    title="Distribution of expert bottleneck norms (log₁₀)",
    xaxis_title="log₁₀(||z||)",
    yaxis_title="Count",
    height=400,
    width=800,
)

fig_dist.write_html("expert_norm_distribution.html")
print("Saved: expert_norm_distribution.html")
