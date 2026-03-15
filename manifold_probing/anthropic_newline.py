#!/usr/bin/env python3
"""
Newline-position manifold analysis with SMIXAE expert activations
using multivariate Fisher discriminant scores.

Pipeline
--------
1. Stream texts from HuggingFace, wrap to fixed line width, tokenize,
   compute per-token ``chars_since_nl``.
2. Collect hidden states from a TransformerLens model at a specified hook point.
3. Encode hidden states through a pretrained SMIXAE (loaded via sae_lens) to
   obtain per-expert bottleneck activations  (n_experts × d_bottleneck).
4. Bucket ``chars_since_nl`` into N equal-width bins, compute the multivariate
   Fisher discriminant ratio  tr(S_W⁻¹ S_B) per expert.
5. Rank experts, visualise the top-10 bottleneck manifolds, and save results.
"""

import gc
import json
import os
import re
import textwrap
from itertools import islice
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import typer
from accelerate.utils import set_seed
from datasets import Dataset, load_dataset
from loguru import logger
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import DataCollatorWithPadding

from sae_lens import SAE

# Ensure SMIXAE architecture is registered with sae_lens
from sae_lens.saes.smixae import SMIXAE  # noqa: F401

# ═══════════════════════ Memory helpers ══════════════════════════════════


def gpu_mem_mb() -> str:
    if not torch.cuda.is_available():
        return ""
    cur = torch.cuda.memory_allocated() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    return f"[GPU {cur:.0f}/{peak:.0f} MB]"


def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ═══════════════════════ Hook helpers ════════════════════════════════════


def extract_layer_from_hook(hook_name: str) -> int | None:
    """Extract layer index from hook name like 'blocks.20.hook_resid_post'."""
    match = re.search(r"blocks\.(\d+)\.", hook_name)
    if match:
        return int(match.group(1))
    return None


# ═══════════════════════ Bucketing ═══════════════════════════════════════


def bucket_labels(
    labels: torch.Tensor, n_buckets: int
) -> tuple[torch.Tensor, np.ndarray]:
    lo = float(labels.min().item())
    hi = float(labels.max().item())
    bin_edges = np.linspace(lo, hi, n_buckets + 1)
    boundaries = torch.tensor(bin_edges[1:-1], dtype=labels.dtype)
    bucketed = torch.bucketize(labels.float(), boundaries.float()).long()
    return bucketed, bin_edges


# ═══════════════════════ Data Pipeline ═══════════════════════════════════


def wrap_preserve_newlines(text: str, width: int, **kw) -> str:
    wrapper = textwrap.TextWrapper(width=width, **kw)
    out: list[str] = []
    for line in text.splitlines(keepends=False):
        if line.strip() == "":
            out.append(line)
        else:
            out.extend(wrapper.wrap(line))
    return "\n".join(out)


def make_line_wrapper(
    line_length: int, text_key: str = "text", out_key: str = "text_lines"
):
    def _fn(ex):
        ex[out_key] = wrap_preserve_newlines(ex[text_key], width=line_length)
        return ex

    return _fn


def assert_chars_since_nl_map(line_length: int, key: str = "chars_since_nl"):
    def _fn(batch):
        bad = [
            (i, m)
            for i, seq in enumerate(batch[key])
            if (m := max(seq, default=0)) > line_length
        ]
        assert not bad, (
            f"chars_since_nl > {line_length} in {len(bad)} seqs; first={bad[0]}"
        )
        return batch

    return _fn


def make_forward_inputs_with_chars_since_nl(
    tokenizer,
    max_seq_len: int,
    use_chat: bool,
    text_key: str = "text_lines",
    out_chars_key: str = "chars_since_nl",
    add_generation_prompt: bool = False,
):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Requires a *fast* tokenizer (for offset_mapping).")

    special_ids = set(getattr(tokenizer, "all_special_ids", ()))

    def _render_and_span(x: str) -> tuple[str, int, int]:
        if not use_chat:
            return x, 0, len(x)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": x}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        start = rendered.find(x)
        if start == -1:
            return rendered, 0, 0
        return rendered, start, start + len(x)

    def _fn(batch: dict[str, Any]) -> dict[str, Any]:
        xs: list[str] = batch[text_key]
        rendered_list, spans = [], []
        for x in xs:
            r, s, e = _render_and_span(x)
            rendered_list.append(r)
            spans.append((s, e))

        enc = tokenizer(
            rendered_list,
            add_special_tokens=not use_chat,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        out_chars: list[list[int]] = []
        for x, (span_s, span_e), ids, offs, spmask in zip(
            xs,
            spans,
            enc["input_ids"],
            enc["offset_mapping"],
            enc["special_tokens_mask"],
        ):
            last_nl = -1
            cols: list[int] = []
            for tid, (s, e), is_sp in zip(ids, offs, spmask):
                tid = int(tid)
                if is_sp or tid in special_ids or not (span_s <= s and e <= span_e):
                    cols.append(0)
                    continue
                s0, e0 = s - span_s, e - span_s
                nl = x.rfind("\n", s0, e0)
                if nl != -1:
                    last_nl = nl
                cols.append(e0 - last_nl - 1)
            out_chars.append(cols)

        enc[out_chars_key] = out_chars
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            out_chars_key: enc[out_chars_key],
        }

    return _fn


# ═══════════════════════ Hidden State Collection ═════════════════════════


def collect_hook_hiddens(
    dataset,
    model: HookedTransformer,
    tokenizer,
    hook_name: str,
    batch_size: int,
    num_workers: int = 0,
    max_seq_len: int | None = None,
) -> list[torch.Tensor]:
    """
    Collect activations at *hook_name* using TransformerLens run_with_cache.

    Uses ``names_filter`` to cache only the requested hook and
    ``stop_at_layer`` to skip computing unnecessary later layers.
    Every batch's GPU tensors are freed after copying the result to CPU.
    """
    model.eval()
    device = next(model.parameters()).device

    # Stop early if possible — no need to compute layers after our hook
    hook_layer = extract_layer_from_hook(hook_name)
    stop_at_layer = hook_layer + 1 if hook_layer is not None else None

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length" if max_seq_len else "longest",
        max_length=max_seq_len,
        return_tensors="pt",
    )

    def collate_fn(examples):
        feats = []
        for ex in examples:
            ids = ex["input_ids"]
            am = ex["attention_mask"]
            feats.append(
                {
                    "input_ids": ids.tolist() if torch.is_tensor(ids) else ids,
                    "attention_mask": am.tolist() if torch.is_tensor(am) else am,
                }
            )
        batch = collator(feats)
        batch["lengths"] = batch["attention_mask"].sum(dim=1, dtype=torch.long)
        return batch

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    layer_out: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(dl, desc=f"Collecting hiddens ({hook_name})"):
            lengths = batch.pop("lengths").tolist()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            _, cache = model.run_with_cache(
                input_ids,
                attention_mask=attention_mask,
                names_filter=hook_name,
                stop_at_layer=stop_at_layer,
                prepend_bos=False,  # we already tokenized with BOS
                return_type=None,  # don't compute logits — saves memory
            )

            h = cache[hook_name].detach().cpu()  # (B, T, D)

            # Free all GPU memory from this batch
            del cache, input_ids, attention_mask, batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

            for i, L in enumerate(lengths):
                if tokenizer.padding_side == "left":
                    layer_out.append(h[i, -L:].contiguous())
                else:
                    layer_out.append(h[i, :L].contiguous())

            del h

    assert len(layer_out) == len(dataset)
    logger.info(f"Collected {len(layer_out)} samples at {hook_name}  {gpu_mem_mb()}")
    return layer_out


# ═══════════════════ SMIXAE Encoding ═════════════════════════════════════


def extract_expert_bottleneck_acts(
    sae: SAE,
    all_hiddens: torch.Tensor,
    batch_size: int = 4096,
) -> torch.Tensor:
    """
    Encode via ``sae.encode()`` in batches. For SMIXAE this returns
    ``(batch, n_experts, d_bottleneck)`` — thresholded bottleneck activations.
    """
    device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    chunks: list[torch.Tensor] = []

    sae.eval()
    with torch.no_grad():
        for batch_cpu in tqdm(all_hiddens.split(batch_size), desc="SMIXAE encode"):
            batch_gpu = batch_cpu.to(device=device, dtype=sae_dtype)
            encoded = sae.encode(batch_gpu)
            chunks.append(encoded.float().cpu())

            del batch_gpu, encoded
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return torch.cat(chunks, dim=0)


# ═══════════════════ Multivariate Fisher Score ═══════════════════════════


def compute_expert_fisher_scores(
    expert_acts: torch.Tensor,  # (N, n_experts, d_bottleneck) — CPU
    labels: torch.Tensor,  # (N,) — CPU, bucketed
    min_samples: int = 5,
) -> np.ndarray:
    """Entirely CPU-based — no GPU memory pressure."""
    N, n_experts, d = expert_acts.shape

    unique_labels, counts = torch.unique(labels, return_counts=True)
    valid = unique_labels[counts >= min_samples]

    if len(valid) < 2:
        logger.warning(
            f"Only {len(valid)} class(es) ≥ {min_samples} samples; need ≥ 2."
        )
        return np.zeros(n_experts, dtype=np.float64)

    logger.info(
        f"Fisher: {len(valid)} classes with ≥ {min_samples} samples "
        f"(counts: {dict(zip(valid.tolist(), counts[counts >= min_samples].tolist()))})"
    )

    mask = torch.isin(labels, valid)
    acts = expert_acts[mask].float()
    labs = labels[mask]

    global_mean = acts.mean(dim=0).double()

    S_B = torch.zeros(n_experts, d, d, dtype=torch.float64)
    S_W = torch.zeros(n_experts, d, d, dtype=torch.float64)

    for c in tqdm(valid.tolist(), desc="Fisher scatter matrices"):
        c_mask = labs == c
        x_c = acts[c_mask].double()
        n_c = x_c.shape[0]
        mean_c = x_c.mean(dim=0)

        diff = (mean_c - global_mean).unsqueeze(-1)
        S_B += n_c * (diff @ diff.transpose(-1, -2))

        centered = x_c - mean_c.unsqueeze(0)
        S_W += torch.einsum("cnd,cne->nde", centered, centered)

        del x_c, centered

    S_W += 1e-8 * torch.eye(d, dtype=torch.float64).unsqueeze(0)

    try:
        X = torch.linalg.solve(S_W, S_B)
        scores = torch.diagonal(X, dim1=-2, dim2=-1).sum(dim=-1)
    except Exception as exc:
        logger.warning(f"Fisher solve failed ({exc}); returning zeros.")
        scores = torch.zeros(n_experts, dtype=torch.float64)

    return scores.numpy()


def compute_expert_class_stats(
    expert_acts: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class stats using the *original* (unbucketed) labels."""
    unique = torch.unique(labels).tolist()
    norms = expert_acts.norm(dim=-1)

    means, rates = [], []
    for c in unique:
        m = labels == c
        means.append(expert_acts[m].float().mean(dim=0).numpy())
        rates.append((norms[m] > threshold).float().mean(dim=0).numpy())

    return np.stack(means), np.stack(rates), np.array(unique)


# ═══════════════════════ Visualisation ═══════════════════════════════════


def plot_top_experts(
    expert_acts: torch.Tensor,
    labels: torch.Tensor,
    fisher_scores: np.ndarray,
    top_k: int = 10,
    max_points: int = 50_000,
    output_path: str = "top_experts.html",
) -> go.Figure:
    """Scatter plots coloured by *original* (continuous) chars_since_nl."""
    N, n_experts, d = expert_acts.shape
    ranking = np.argsort(fisher_scores)[::-1][:top_k].copy()

    if max_points < N:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_points, replace=False)
        idx.sort()
    else:
        idx = np.arange(N)

    labels_np = labels[idx].numpy().astype(np.float32)
    vmin, vmax = float(labels_np.min()), float(labels_np.max())

    use_3d = d >= 3
    nrows, ncols = 2, 5

    subplot_titles = [f"Expert {eid} — J={fisher_scores[eid]:.2f}" for eid in ranking]
    subplot_titles += [""] * (nrows * ncols - len(subplot_titles))

    if use_3d:
        specs = [[{"type": "scene"} for _ in range(ncols)] for _ in range(nrows)]
    else:
        specs = [[{"type": "xy"} for _ in range(ncols)] for _ in range(nrows)]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        specs=specs,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for plot_i, expert_id in enumerate(ranking):
        row = plot_i // ncols + 1
        col = plot_i % ncols + 1

        acts_e = expert_acts[idx, expert_id].numpy()

        marker_kwargs = dict(
            color=labels_np,
            colorscale="Viridis",
            cmin=vmin,
            cmax=vmax,
            size=1.5,
            opacity=0.6,
        )
        if plot_i == len(ranking) - 1:
            marker_kwargs["colorbar"] = dict(
                title="chars<br>since \\n",
                len=0.4,
                thickness=15,
                x=1.02,
                y=0.5,
            )
            marker_kwargs["showscale"] = True
        else:
            marker_kwargs["showscale"] = False

        if use_3d:
            trace = go.Scatter3d(
                x=acts_e[:, 0],
                y=acts_e[:, 1],
                z=acts_e[:, 2],
                mode="markers",
                marker=marker_kwargs,
                hovertemplate=(
                    "b0: %{x:.3f}<br>b1: %{y:.3f}<br>b2: %{z:.3f}<br>"
                    "chars_since_nl: %{marker.color:.0f}<extra></extra>"
                ),
                name=f"Expert {expert_id}",
                showlegend=False,
            )
            fig.add_trace(trace, row=row, col=col)

            scene_name = f"scene{plot_i + 1}" if plot_i > 0 else "scene"
            fig.update_layout(
                **{
                    scene_name: dict(
                        xaxis_title="b₀",
                        yaxis_title="b₁",
                        zaxis_title="b₂",
                        aspectmode="data",
                        xaxis=dict(
                            showbackground=False,
                            showticklabels=False,
                            title_font_size=10,
                        ),
                        yaxis=dict(
                            showbackground=False,
                            showticklabels=False,
                            title_font_size=10,
                        ),
                        zaxis=dict(
                            showbackground=False,
                            showticklabels=False,
                            title_font_size=10,
                        ),
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                    )
                }
            )
        else:
            trace = go.Scattergl(
                x=acts_e[:, 0],
                y=acts_e[:, 1] if d >= 2 else np.zeros_like(acts_e[:, 0]),
                mode="markers",
                marker=marker_kwargs,
                hovertemplate=(
                    "b0: %{x:.3f}<br>b1: %{y:.3f}<br>"
                    "chars_since_nl: %{marker.color:.0f}<extra></extra>"
                ),
                name=f"Expert {expert_id}",
                showlegend=False,
            )
            fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        title=dict(
            text=f"Top-{top_k} SMIXAE Experts — Bottleneck Activations by chars_since_nl",
            font_size=16,
        ),
        height=800,
        width=2000,
        template="plotly_white",
        margin=dict(l=20, r=60, t=60, b=20),
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    logger.info(f"Saved scatter plot → {output_path}")
    return fig


def plot_expert_class_means(
    class_means: np.ndarray,
    class_labels: np.ndarray,
    fisher_scores: np.ndarray,
    top_k: int = 10,
    output_path: str = "top_experts_class_means.html",
) -> go.Figure:
    """Class-mean trajectories coloured by original chars_since_nl."""
    n_classes, n_experts, d = class_means.shape
    ranking = np.argsort(fisher_scores)[::-1][:top_k].copy()

    labels_float = class_labels.astype(np.float32)
    vmin, vmax = float(labels_float.min()), float(labels_float.max())

    use_3d = d >= 3
    nrows, ncols = 2, 5

    subplot_titles = [f"Expert {eid} — J={fisher_scores[eid]:.2f}" for eid in ranking]
    subplot_titles += [""] * (nrows * ncols - len(subplot_titles))

    if use_3d:
        specs = [[{"type": "scene"} for _ in range(ncols)] for _ in range(nrows)]
    else:
        specs = [[{"type": "xy"} for _ in range(ncols)] for _ in range(nrows)]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        specs=specs,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for plot_i, expert_id in enumerate(ranking):
        row = plot_i // ncols + 1
        col = plot_i % ncols + 1

        m = class_means[:, expert_id, :]

        marker_kwargs = dict(
            color=labels_float,
            colorscale="Viridis",
            cmin=vmin,
            cmax=vmax,
            size=5,
            opacity=0.9,
            line=dict(width=0.5, color="black"),
        )
        if plot_i == len(ranking) - 1:
            marker_kwargs["colorbar"] = dict(
                title="chars<br>since \\n",
                len=0.4,
                thickness=15,
                x=1.02,
                y=0.5,
            )
            marker_kwargs["showscale"] = True
        else:
            marker_kwargs["showscale"] = False

        if use_3d:
            fig.add_trace(
                go.Scatter3d(
                    x=m[:, 0],
                    y=m[:, 1],
                    z=m[:, 2],
                    mode="lines",
                    line=dict(color="rgba(100,100,100,0.3)", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=m[:, 0],
                    y=m[:, 1],
                    z=m[:, 2],
                    mode="markers",
                    marker=marker_kwargs,
                    hovertemplate=(
                        "b0: %{x:.3f}<br>b1: %{y:.3f}<br>b2: %{z:.3f}<br>"
                        "chars_since_nl: %{marker.color:.0f}<extra></extra>"
                    ),
                    name=f"Expert {expert_id}",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            scene_name = f"scene{plot_i + 1}" if plot_i > 0 else "scene"
            fig.update_layout(
                **{
                    scene_name: dict(
                        xaxis_title="b₀",
                        yaxis_title="b₁",
                        zaxis_title="b₂",
                        aspectmode="data",
                        xaxis=dict(
                            showbackground=False,
                            showticklabels=False,
                            title_font_size=10,
                        ),
                        yaxis=dict(
                            showbackground=False,
                            showticklabels=False,
                            title_font_size=10,
                        ),
                        zaxis=dict(
                            showbackground=False,
                            showticklabels=False,
                            title_font_size=10,
                        ),
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                    )
                }
            )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=m[:, 0],
                    y=m[:, 1] if d >= 2 else np.zeros(n_classes),
                    mode="markers+lines",
                    marker=marker_kwargs,
                    line=dict(color="rgba(100,100,100,0.3)", width=1),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=dict(
            text=f"Top-{top_k} Experts — Class Mean Trajectories by chars_since_nl",
            font_size=16,
        ),
        height=800,
        width=2000,
        template="plotly_white",
        margin=dict(l=20, r=60, t=60, b=20),
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    logger.info(f"Saved class-mean plot → {output_path}")
    return fig


# ═══════════════════════════ CLI ═════════════════════════════════════════

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    # Model & data
    model_name: str = typer.Option(
        "google/gemma-2-9b", help="HuggingFace model name or path."
    ),
    dataset_name: str = typer.Option(
        "monology/pile-uncopyrighted", help="HuggingFace dataset identifier."
    ),
    output_path: str = typer.Option("outputs", help="Root output directory."),
    batch_size: int = typer.Option(8, help="Batch size for LLM forward passes."),
    num_workers: int = typer.Option(0, help="DataLoader workers."),
    # Text processing
    line_length: int = typer.Option(
        80, help="Wrap text to this many characters per line."
    ),
    num_samples: int = typer.Option(100, help="Number of post-filter samples."),
    min_lines: int = typer.Option(
        5, help="Reject texts shorter than line_length × min_lines."
    ),
    max_seq_len: int = typer.Option(2048, help="Tokenizer truncation limit."),
    use_chat: bool = typer.Option(False, help="Apply chat template before tokenizing."),
    # SMIXAE
    smixae_path: str = typer.Option(
        ..., help="Path to SMIXAE checkpoint dir (sae_lens format)."
    ),
    hook_name: str = typer.Option(
        "blocks.20.hook_resid_post",
        help="TransformerLens hook point the SMIXAE was trained on (e.g. blocks.20.hook_resid_post).",
    ),
    sae_batch_size: int = typer.Option(4096, help="Batch size for SMIXAE encoding."),
    # Fisher
    fisher_n_buckets: int = typer.Option(
        5, help="Number of equal-width bins for chars_since_nl for Fisher."
    ),
    fisher_min_samples_per_class: int = typer.Option(
        5, help="Min samples per bucket for Fisher."
    ),
    fisher_top_k: int = typer.Option(50, help="How many top experts to log."),
    # Visualisation
    plot_top_k: int = typer.Option(10, help="How many top experts to plot."),
    plot_max_points: int = typer.Option(
        50_000, help="Subsample scatter plots to this many points."
    ),
    # Misc
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Analyse SMIXAE experts for newline-position manifold structure."""

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    # ── Output dir ───────────────────────────────────────────────────────
    model_slug = os.path.basename(model_name.strip("/"))
    ds_slug = os.path.basename(dataset_name.strip("/"))
    out_dir = os.path.join(output_path, model_slug, ds_slug)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model via TransformerLens ────────────────────────────────────
    if not any(k in model_name for k in ("gpt2", "pythia")):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    logger.info(f"Loading model {model_name} via TransformerLens  {gpu_mem_mb()}")
    model = HookedTransformer.from_pretrained(
        model_name,
        dtype=dtype,
        device=str(device),
    )
    model.eval()
    logger.info(
        f"Model loaded ({model.cfg.n_layers} layers, n_ctx={model.cfg.n_ctx})  {gpu_mem_mb()}"
    )

    # Validate hook name
    valid_hooks = set(model.hook_dict.keys())
    if hook_name not in valid_hooks:
        logger.error(
            f"Hook '{hook_name}' not found. "
            f"Available hooks containing 'resid': "
            f"{sorted(h for h in valid_hooks if 'resid' in h)[:20]}"
        )
        raise typer.BadParameter(f"Unknown hook: {hook_name}")

    hook_layer = extract_layer_from_hook(hook_name)
    logger.info(f"Using hook: {hook_name}  (layer={hook_layer})")

    # ── Tokenizer (from TransformerLens) ─────────────────────────────────
    tokenizer = model.tokenizer
    assert tokenizer is not None, "TransformerLens model has no tokenizer"

    tokenizer.padding_side = "left" if "Qwen3" in model_name else "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve max_seq_len against model context window
    model_max = model.cfg.n_ctx
    assert 0 < max_seq_len <= model_max, (
        f"max_seq_len={max_seq_len} must be in (0, {model_max}]"
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    logger.info(
        f"Streaming {dataset_name} (filter: len>{line_length * min_lines}, no \\n)"
    )
    stream = load_dataset(dataset_name, split="train", streaming=True)
    stream = stream.filter(
        lambda x: len(x["text"]) > line_length * min_lines and "\n" not in x["text"]
    )
    dataset = Dataset.from_generator(lambda: islice(stream, num_samples))
    logger.info(f"Materialised {len(dataset)} samples")

    dataset = dataset.map(
        make_line_wrapper(line_length),
        desc="Wrapping into char-limited lines",
    )
    dataset = dataset.map(
        make_forward_inputs_with_chars_since_nl(
            tokenizer,
            max_seq_len,
            use_chat=use_chat,
            text_key="text_lines",
        ),
        batched=True,
        desc="Tokenizing + computing chars_since_nl",
    )
    dataset = dataset.with_format(
        "torch",
        columns=[
            c for c in dataset.column_names if c in ("input_ids", "attention_mask")
        ],
        output_all_columns=True,
    )
    dataset = dataset.map(
        assert_chars_since_nl_map(line_length),
        batched=True,
        desc="Validating chars_since_nl",
    )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Collect hidden states at hook, then FREE the model
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"Collecting hidden states at {hook_name}  {gpu_mem_mb()}")
    layer_hiddens = collect_hook_hiddens(
        dataset,
        model,
        tokenizer,
        hook_name=hook_name,
        batch_size=batch_size,
        num_workers=num_workers,
        max_seq_len=max_seq_len,
    )

    # Flatten → CPU
    all_hiddens = torch.cat(layer_hiddens, dim=0)
    all_labels = torch.cat(
        [torch.as_tensor(c, dtype=torch.long) for c in dataset["chars_since_nl"]]
    )
    assert all_hiddens.shape[0] == all_labels.shape[0]

    keep = all_labels > 0
    all_hiddens = all_hiddens[keep]
    all_labels = all_labels[keep]
    logger.info(f"Tokens after dropping label==0: {all_hiddens.shape[0]:,}")

    # Free base model — reclaim all GPU memory
    del model, layer_hiddens
    flush_gpu()
    logger.info(f"Base model freed  {gpu_mem_mb()}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Load SMIXAE via sae_lens, encode, then FREE it
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"Loading SMIXAE from {smixae_path}  {gpu_mem_mb()}")
    sae = SAE.load_from_disk(path=smixae_path, device=str(device))
    sae.eval()

    logger.info(
        f"SMIXAE loaded: {sae.cfg.n_experts} experts, "
        f"d_expert={sae.cfg.d_expert}, d_bottleneck={sae.cfg.d_bottleneck}  "
        f"{gpu_mem_mb()}"
    )

    expert_acts = extract_expert_bottleneck_acts(
        sae,
        all_hiddens,
        batch_size=sae_batch_size,
    )
    logger.info(f"Expert activations: {expert_acts.shape}  {gpu_mem_mb()}")

    threshold = float(sae.threshold.item())
    n_experts = sae.cfg.n_experts
    d_bottleneck = sae.cfg.d_bottleneck

    del sae, all_hiddens
    flush_gpu()
    logger.info(f"SMIXAE freed  {gpu_mem_mb()}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Fisher scores + visualisation — all CPU
    # ══════════════════════════════════════════════════════════════════════

    # Bucket labels for Fisher
    bucketed_labels, bin_edges = bucket_labels(all_labels, fisher_n_buckets)

    bucket_counts = torch.bincount(bucketed_labels)
    logger.info(
        f"Bucketed chars_since_nl into {fisher_n_buckets} bins:\n"
        + "\n".join(
            f"  bucket {i}: [{bin_edges[i]:.0f}, {bin_edges[i + 1]:.0f})  "
            f"n={bucket_counts[i].item():,}"
            for i in range(fisher_n_buckets)
        )
    )

    fisher_scores = compute_expert_fisher_scores(
        expert_acts,
        bucketed_labels,
        min_samples=fisher_min_samples_per_class,
    )

    ranking = np.argsort(fisher_scores)[::-1].copy()

    logger.info(f"\n{'═' * 60}")
    logger.info(f"Top-{fisher_top_k} experts by Fisher score ({hook_name}):")
    for i, eidx in enumerate(ranking[:fisher_top_k]):
        logger.info(f"  #{i + 1:3d}  expert {eidx:4d}  J = {fisher_scores[eidx]:.4f}")

    # Per-class stats use original labels
    class_means, firing_rates, class_labels_arr = compute_expert_class_stats(
        expert_acts,
        all_labels,
        threshold=threshold,
    )

    # Plots use original labels for continuous Viridis colouring
    plot_top_experts(
        expert_acts=expert_acts,
        labels=all_labels,
        fisher_scores=fisher_scores,
        top_k=plot_top_k,
        max_points=plot_max_points,
        output_path=os.path.join(out_dir, "top_experts_scatter.html"),
    )

    plot_expert_class_means(
        class_means=class_means,
        class_labels=class_labels_arr,
        fisher_scores=fisher_scores,
        top_k=plot_top_k,
        output_path=os.path.join(out_dir, "top_experts_class_means.html"),
    )

    # ── Save numerical results ───────────────────────────────────────────
    fisher_df = pd.DataFrame(
        {
            "expert_id": np.arange(len(fisher_scores)),
            "fisher_score": fisher_scores,
        }
    )
    fisher_df = (
        fisher_df.sort_values("fisher_score", ascending=False)
        .reset_index(drop=True)
        .rename_axis("rank")
    )
    fisher_df.to_csv(os.path.join(out_dir, "fisher_scores.csv"))

    np.save(os.path.join(out_dir, "fisher_scores.npy"), fisher_scores)
    np.save(os.path.join(out_dir, "expert_class_means.npy"), class_means)
    np.save(os.path.join(out_dir, "expert_firing_rates.npy"), firing_rates)
    np.save(os.path.join(out_dir, "class_labels.npy"), class_labels_arr)

    n_tokens = int(all_labels.shape[0])
    summary = {
        "hook_name": hook_name,
        "hook_layer": hook_layer,
        "n_experts": n_experts,
        "d_bottleneck": d_bottleneck,
        "n_tokens": n_tokens,
        "n_classes_original": int(len(class_labels_arr)),
        "fisher_n_buckets": fisher_n_buckets,
        "fisher_bin_edges": bin_edges.tolist(),
        "fisher_mean": float(fisher_scores.mean()),
        "fisher_std": float(fisher_scores.std()),
        "fisher_max": float(fisher_scores.max()),
        "fisher_median": float(np.median(fisher_scores)),
        "fisher_top1_expert": int(ranking[0]),
        "fisher_top1_score": float(fisher_scores[ranking[0]]),
        "fisher_top5_mean": float(fisher_scores[ranking[:5]].mean()),
        "fisher_top10_mean": float(fisher_scores[ranking[:10]].mean()),
        "threshold": threshold,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    logger.info(f"\nResults saved to {out_dir}")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    app()
