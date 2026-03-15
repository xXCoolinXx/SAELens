#!/usr/bin/env python3
"""
Newline-position manifold analysis with SMIXAE expert activations.

Scores each expert with four metrics:
  • decode_r2:          R² of  chars_since_nl ~ linear(bottleneck)
  • encode_linear_r2:   mean R² of  bottleneck_dim ~ linear(chars)
  • encode_periodic_r2: mean R² of  bottleneck_dim ~ linear(chars) + Fourier(chars)
  • periodic_gain:      encode_periodic_r2 − encode_linear_r2

Generates HTML plots showing the top-k experts for EVERY method side by side.
"""

import gc
import json
import math
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
from sae_lens.saes.smixae import SMIXAE  # noqa: F401 — registers architecture

# ═══════════════════════ Constants ═══════════════════════════════════════

VALID_RANK_BY = [
    "decode_r2",
    "encode_linear_r2",
    "encode_periodic_r2",
    "periodic_gain",
]
METHOD_LABELS = {
    "decode_r2": "Decode R²",
    "encode_linear_r2": "Linear R²",
    "encode_periodic_r2": "Periodic R²",
    "periodic_gain": "Periodic Δ",
}
METHOD_SHORT = {
    "decode_r2": "dec",
    "encode_linear_r2": "lin",
    "encode_periodic_r2": "per",
    "periodic_gain": "Δper",
}


# ═══════════════════════ Memory ══════════════════════════════════════════


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


# ═══════════════════════ Helpers ═════════════════════════════════════════


def extract_layer_from_hook(hook_name: str) -> int | None:
    m = re.search(r"blocks\.(\d+)\.", hook_name)
    return int(m.group(1)) if m else None


def _grid_layout(n_methods: int, top_k: int, max_cols: int = 5):
    """Returns (total_rows, ncols, rows_per_method)."""
    ncols = min(top_k, max_cols)
    rows_per = max(math.ceil(top_k / ncols), 1) if ncols > 0 else 1
    return rows_per * n_methods, ncols, rows_per


# ═══════════════════════ Data Pipeline ═══════════════════════════════════


def wrap_preserve_newlines(text: str, width: int) -> str:
    wrapper = textwrap.TextWrapper(width=width)
    out: list[str] = []
    for line in text.splitlines(keepends=False):
        if line.strip() == "":
            out.append(line)
        else:
            out.extend(wrapper.wrap(line))
    return "\n".join(out)


def make_line_wrapper(line_length: int):
    def _fn(ex):
        ex["text_lines"] = wrap_preserve_newlines(ex["text"], width=line_length)
        return ex

    return _fn


def assert_chars_since_nl_map(line_length: int):
    def _fn(batch):
        bad = [
            (i, m)
            for i, seq in enumerate(batch["chars_since_nl"])
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
):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Requires a *fast* tokenizer (for offset_mapping).")
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))

    def _render(x: str):
        if not use_chat:
            return x, 0, len(x)
        r = tokenizer.apply_chat_template(
            [{"role": "user", "content": x}],
            tokenize=False,
            add_generation_prompt=False,
        )
        s = r.find(x)
        return (r, s, s + len(x)) if s != -1 else (r, 0, 0)

    def _fn(batch: dict[str, Any]) -> dict[str, Any]:
        xs = batch["text_lines"]
        rendered_list, spans = [], []
        for x in xs:
            r, s, e = _render(x)
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
            last_nl, cols = -1, []
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

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "chars_since_nl": out_chars,
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
    model.eval()
    device = next(model.parameters()).device
    hook_layer = extract_layer_from_hook(hook_name)
    stop_at = hook_layer + 1 if hook_layer is not None else None

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length" if max_seq_len else "longest",
        max_length=max_seq_len,
        return_tensors="pt",
    )

    def collate_fn(examples):
        feats = [
            {
                "input_ids": (
                    e["input_ids"].tolist()
                    if torch.is_tensor(e["input_ids"])
                    else e["input_ids"]
                ),
                "attention_mask": (
                    e["attention_mask"].tolist()
                    if torch.is_tensor(e["attention_mask"])
                    else e["attention_mask"]
                ),
            }
            for e in examples
        ]
        b = collator(feats)
        b["lengths"] = b["attention_mask"].sum(dim=1, dtype=torch.long)
        return b

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    out: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(dl, desc=f"Hiddens ({hook_name})"):
            lengths = batch.pop("lengths").tolist()
            ids = batch["input_ids"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)

            _, cache = model.run_with_cache(
                ids,
                attention_mask=am,
                names_filter=hook_name,
                stop_at_layer=stop_at,
                prepend_bos=False,
                return_type=None,
            )
            h = cache[hook_name].detach().cpu()

            del cache, ids, am, batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

            for i, L in enumerate(lengths):
                if tokenizer.padding_side == "left":
                    out.append(h[i, -L:].contiguous())
                else:
                    out.append(h[i, :L].contiguous())
            del h

    assert len(out) == len(dataset)
    logger.info(f"Collected {len(out)} samples  {gpu_mem_mb()}")
    return out


# ═══════════════════ SMIXAE Encoding ═════════════════════════════════════


def extract_expert_bottleneck_acts(
    sae: SAE,
    all_hiddens: torch.Tensor,
    batch_size: int = 4096,
) -> torch.Tensor:
    device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    chunks: list[torch.Tensor] = []

    sae.eval()
    with torch.no_grad():
        for b_cpu in tqdm(all_hiddens.split(batch_size), desc="SMIXAE encode"):
            b_gpu = b_cpu.to(device=device, dtype=sae_dtype)
            chunks.append(sae.encode(b_gpu).float().cpu())
            del b_gpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return torch.cat(chunks, dim=0)


# ═══════════════════ Regression Scoring ══════════════════════════════════


def compute_expert_scores(
    expert_acts: torch.Tensor,  # (N, n_experts, d_bottleneck) CPU
    labels: torch.Tensor,  # (N,) CPU
    line_length: int,
    n_harmonics: int = 3,
    chunk_size: int = 64,
) -> pd.DataFrame:
    N, n_experts, d = expert_acts.shape
    chars = labels.double()

    # Build design matrices
    ones = torch.ones(N, dtype=torch.float64)
    X_lin = torch.stack([chars, ones], dim=-1)  # (N, 2)

    feats = [chars, ones]
    for k in range(1, n_harmonics + 1):
        phase = 2 * torch.pi * k * chars / line_length
        feats.extend([torch.sin(phase), torch.cos(phase)])
    X_per = torch.stack(feats, dim=-1)  # (N, 2+2H)

    X_lin_pinv = torch.linalg.pinv(X_lin)  # (2, N)
    X_per_pinv = torch.linalg.pinv(X_per)  # (F, N)

    enc_lin_r2 = torch.zeros(n_experts, d, dtype=torch.float64)
    enc_per_r2 = torch.zeros(n_experts, d, dtype=torch.float64)
    dec_r2 = torch.zeros(n_experts, dtype=torch.float64)
    corrs = torch.zeros(n_experts, d, dtype=torch.float64)

    chars_c = chars - chars.mean()
    ss_tot_chars = (chars_c**2).sum().clamp(min=1e-10)

    for start in tqdm(range(0, n_experts, chunk_size), desc="Expert scores"):
        end = min(start + chunk_size, n_experts)
        nc = end - start

        # Reshape chunk of expert activations to (N, nc*d)
        Y = expert_acts[:, start:end, :].double().reshape(N, nc * d)
        Y_mean = Y.mean(dim=0, keepdim=True)
        ss_tot = ((Y - Y_mean) ** 2).sum(dim=0).clamp(min=1e-10)

        # Encode linear: bottleneck_dim ~ linear(chars)
        Yh_lin = X_lin @ (X_lin_pinv @ Y)
        enc_lin_r2[start:end] = (1 - ((Y - Yh_lin) ** 2).sum(0) / ss_tot).reshape(nc, d)

        # Encode periodic: bottleneck_dim ~ linear(chars) + Fourier(chars)
        Yh_per = X_per @ (X_per_pinv @ Y)
        enc_per_r2[start:end] = (1 - ((Y - Yh_per) ** 2).sum(0) / ss_tot).reshape(nc, d)

        # Decode: chars ~ linear(bottleneck)
        acts_chunk = expert_acts[:, start:end, :].double()  # (N, nc, d)
        X_dec = torch.cat(
            [acts_chunk, torch.ones(N, nc, 1, dtype=torch.float64)],
            dim=-1,
        )  # (N, nc, d+1)
        y_target = chars.view(1, N, 1).expand(nc, -1, -1)  # (nc, N, 1)
        beta = torch.linalg.lstsq(X_dec.permute(1, 0, 2), y_target).solution
        y_hat = (X_dec.permute(1, 0, 2) @ beta).squeeze(-1)  # (nc, N)
        dec_r2[start:end] = (
            1 - ((chars.unsqueeze(0) - y_hat) ** 2).sum(1) / ss_tot_chars
        )

        # Per-dim Pearson correlation
        acts_c = acts_chunk - acts_chunk.mean(dim=0, keepdim=True)
        cov = (chars_c.view(N, 1, 1) * acts_c).sum(dim=0)
        denom = ss_tot_chars.sqrt() * (acts_c**2).sum(dim=0).clamp(min=1e-10).sqrt()
        corrs[start:end] = cov / denom

    # Assemble DataFrame
    rows = []
    for e in range(n_experts):
        r = {
            "expert_id": e,
            "decode_r2": float(dec_r2[e]),
            "encode_linear_r2": float(enc_lin_r2[e].mean()),
            "encode_periodic_r2": float(enc_per_r2[e].mean()),
            "periodic_gain": float(enc_per_r2[e].mean() - enc_lin_r2[e].mean()),
        }
        for j in range(d):
            r[f"dim{j}_corr"] = float(corrs[e, j])
            r[f"dim{j}_linear_r2"] = float(enc_lin_r2[e, j])
            r[f"dim{j}_periodic_r2"] = float(enc_per_r2[e, j])
        rows.append(r)
    return pd.DataFrame(rows)


def compute_expert_class_stats(
    expert_acts: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = torch.unique(labels).tolist()
    norms = expert_acts.norm(dim=-1)
    means, rates = [], []
    for c in unique:
        m = labels == c
        means.append(expert_acts[m].float().mean(dim=0).numpy())
        rates.append((norms[m] > threshold).float().mean(dim=0).numpy())
    return np.stack(means), np.stack(rates), np.array(unique)


# ═══════════════════════ Visualisation ═══════════════════════════════════


def _build_multi_method_grid(
    scores_df: pd.DataFrame,
    top_k: int,
    use_3d: bool,
    max_cols: int = 5,
):
    """
    Build subplot grid and collect per-method expert lists.

    Returns (fig, total_rows, ncols, rows_per, method_eids, titles).
    """
    methods = VALID_RANK_BY
    actual_k = min(top_k, len(scores_df))
    total_rows, ncols, rows_per = _grid_layout(len(methods), actual_k, max_cols)

    titles: list[str] = []
    method_eids: dict[str, list[int]] = {}

    for method in methods:
        ranked = scores_df.sort_values(method, ascending=False).head(actual_k)
        eids: list[int] = []
        for _, row in ranked.iterrows():
            eid = int(row["expert_id"])
            val = float(row[method])
            eids.append(eid)
            titles.append(f"{METHOD_SHORT[method]}: E{eid} ({val:.3f})")
        # Pad remaining cells in this method's row-block
        titles.extend([""] * (rows_per * ncols - len(eids)))
        method_eids[method] = eids

    spec_type = "scene" if use_3d else "xy"
    specs = [[{"type": spec_type} for _ in range(ncols)] for _ in range(total_rows)]

    fig = make_subplots(
        rows=total_rows,
        cols=ncols,
        subplot_titles=titles,
        specs=specs,
        horizontal_spacing=0.02,
        vertical_spacing=0.025,
    )

    return fig, total_rows, ncols, rows_per, method_eids


def _add_method_annotations(fig, methods, rows_per, total_rows):
    """Add rotated method labels on the left margin."""
    for m_i, method in enumerate(methods):
        y = 1.0 - (m_i * rows_per + rows_per / 2) / total_rows
        fig.add_annotation(
            text=f"<b>{METHOD_LABELS[method]}</b>",
            xref="paper",
            yref="paper",
            x=-0.05,
            y=y,
            showarrow=False,
            font=dict(size=12),
            textangle=-90,
        )


def _update_3d_scene(fig, flat_idx: int):
    """Configure a 3D scene by its flat (0-based) grid index."""
    scene_key = "scene" if flat_idx == 0 else f"scene{flat_idx + 1}"
    fig.update_layout(
        **{
            scene_key: dict(
                aspectmode="data",
                xaxis=dict(showbackground=False, showticklabels=False, title=""),
                yaxis=dict(showbackground=False, showticklabels=False, title=""),
                zaxis=dict(showbackground=False, showticklabels=False, title=""),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            )
        }
    )


def plot_multi_method_scatter(
    expert_acts: torch.Tensor,
    labels: torch.Tensor,
    scores_df: pd.DataFrame,
    top_k: int = 10,
    max_points: int = 50_000,
    output_path: str = "scatter.html",
) -> go.Figure:
    """One scatter subplot per top expert, arranged by method (one row-group each)."""
    if len(scores_df) == 0:
        logger.warning("No experts to plot (scatter)")
        return go.Figure()

    methods = VALID_RANK_BY
    N, _, d = expert_acts.shape
    use_3d = d >= 3
    actual_k = min(top_k, len(scores_df))

    fig, total_rows, ncols, rows_per, method_eids = _build_multi_method_grid(
        scores_df,
        actual_k,
        use_3d,
    )

    # Reduce points for large grids
    n_subplots = len(methods) * actual_k
    adj_points = min(max_points, N)
    if n_subplots > 10:
        adj_points = max(adj_points * 10 // n_subplots, 2000)

    if adj_points < N:
        idx = np.sort(np.random.default_rng(0).choice(N, adj_points, replace=False))
    else:
        idx = np.arange(N)

    labels_np = labels[idx].numpy().astype(np.float32)
    vmin, vmax = float(labels_np.min()), float(labels_np.max())

    colorbar_placed = False

    for m_i, method in enumerate(methods):
        eids = method_eids[method]
        for pi, eid in enumerate(eids):
            row = m_i * rows_per + pi // ncols + 1
            col = pi % ncols + 1
            flat = (row - 1) * ncols + (col - 1)
            is_last = m_i == len(methods) - 1 and pi == len(eids) - 1

            acts_e = expert_acts[idx, eid].numpy()
            mk = dict(
                color=labels_np,
                colorscale="Viridis",
                cmin=vmin,
                cmax=vmax,
                size=1.0,
                opacity=0.4,
                showscale=(is_last and not colorbar_placed),
            )
            if is_last and not colorbar_placed:
                mk["colorbar"] = dict(
                    title="chars<br>since \\n",
                    len=0.12,
                    thickness=10,
                    x=1.02,
                )
                colorbar_placed = True

            if use_3d:
                fig.add_trace(
                    go.Scatter3d(
                        x=acts_e[:, 0],
                        y=acts_e[:, 1],
                        z=acts_e[:, 2],
                        mode="markers",
                        marker=mk,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
                _update_3d_scene(fig, flat)
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=acts_e[:, 0],
                        y=acts_e[:, 1] if d >= 2 else np.zeros(len(acts_e)),
                        mode="markers",
                        marker=mk,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

    _add_method_annotations(fig, methods, rows_per, total_rows)

    fig.update_layout(
        title="Top Experts by All Scoring Methods — Bottleneck Scatter",
        height=280 * total_rows + 80,
        width=280 * ncols + 140,
        template="plotly_white",
        margin=dict(l=70, r=80, t=60, b=20),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    logger.info(f"Saved multi-method scatter → {output_path}")
    return fig


def plot_multi_method_class_means(
    class_means: np.ndarray,  # (n_classes, n_experts, d)
    class_labels: np.ndarray,  # (n_classes,)
    scores_df: pd.DataFrame,
    top_k: int = 10,
    output_path: str = "class_means.html",
) -> go.Figure:
    """Class-mean trajectory per expert, arranged by method."""
    if len(scores_df) == 0:
        logger.warning("No experts to plot (class means)")
        return go.Figure()

    methods = VALID_RANK_BY
    n_classes, _, d = class_means.shape
    use_3d = d >= 3
    actual_k = min(top_k, len(scores_df))

    fig, total_rows, ncols, rows_per, method_eids = _build_multi_method_grid(
        scores_df,
        actual_k,
        use_3d,
    )

    lf = class_labels.astype(np.float32)
    vmin, vmax = float(lf.min()), float(lf.max())

    colorbar_placed = False

    for m_i, method in enumerate(methods):
        eids = method_eids[method]
        for pi, eid in enumerate(eids):
            row = m_i * rows_per + pi // ncols + 1
            col = pi % ncols + 1
            flat = (row - 1) * ncols + (col - 1)
            is_last = m_i == len(methods) - 1 and pi == len(eids) - 1

            m = class_means[:, eid, :]
            mk = dict(
                color=lf,
                colorscale="Viridis",
                cmin=vmin,
                cmax=vmax,
                size=2.0,
                opacity=0.9,
                line=dict(width=0.3, color="black"),
                showscale=(is_last and not colorbar_placed),
            )
            if is_last and not colorbar_placed:
                mk["colorbar"] = dict(
                    title="chars<br>since \\n",
                    len=0.12,
                    thickness=10,
                    x=1.02,
                )
                colorbar_placed = True

            if use_3d:
                # Connecting line
                fig.add_trace(
                    go.Scatter3d(
                        x=m[:, 0],
                        y=m[:, 1],
                        z=m[:, 2],
                        mode="lines",
                        line=dict(color="rgba(100,100,100,0.3)", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
                # Markers
                fig.add_trace(
                    go.Scatter3d(
                        x=m[:, 0],
                        y=m[:, 1],
                        z=m[:, 2],
                        mode="markers",
                        marker=mk,
                        showlegend=False,
                        hovertemplate=(
                            "b0:%{x:.3f}<br>b1:%{y:.3f}<br>b2:%{z:.3f}<br>"
                            "chars:%{marker.color:.0f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                _update_3d_scene(fig, flat)
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=m[:, 0],
                        y=m[:, 1] if d >= 2 else np.zeros(n_classes),
                        mode="markers+lines",
                        marker=mk,
                        line=dict(color="rgba(100,100,100,0.3)", width=1),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

    _add_method_annotations(fig, methods, rows_per, total_rows)

    fig.update_layout(
        title="Top Experts by All Methods — Class Mean Trajectories",
        height=300 * total_rows + 80,
        width=280 * ncols + 140,
        template="plotly_white",
        margin=dict(l=70, r=80, t=60, b=20),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    logger.info(f"Saved multi-method class means → {output_path}")
    return fig


def plot_expert_dim_analysis(
    expert_acts: torch.Tensor,
    labels: torch.Tensor,
    expert_id: int,
    scores_row: dict,
    line_length: int,
    n_harmonics: int = 3,
    max_points: int = 20_000,
    output_path: str = "dim_analysis.html",
) -> go.Figure:
    """Per-dimension scatter with fitted linear + periodic curves for one expert."""
    N, _, d = expert_acts.shape

    if max_points < N:
        idx = np.sort(np.random.default_rng(42).choice(N, max_points, replace=False))
    else:
        idx = np.arange(N)

    acts_sub = expert_acts[idx, expert_id, :].float().numpy()
    chars_sub = labels[idx].float().numpy()
    chars_full = labels.double().numpy()
    acts_full = expert_acts[:, expert_id, :].double().numpy()

    x_fit = np.linspace(chars_full.min(), chars_full.max(), 300)

    def _periodic_design(x):
        cols = [x, np.ones_like(x)]
        for k in range(1, n_harmonics + 1):
            ph = 2 * np.pi * k * x / line_length
            cols.extend([np.sin(ph), np.cos(ph)])
        return np.column_stack(cols)

    X_per_full = _periodic_design(chars_full)
    X_per_fit = _periodic_design(x_fit)

    fig = make_subplots(
        rows=1,
        cols=d,
        subplot_titles=[
            f"Dim {j}  (ρ={scores_row.get(f'dim{j}_corr', 0):.3f}, "
            f"lin={scores_row.get(f'dim{j}_linear_r2', 0):.3f}, "
            f"per={scores_row.get(f'dim{j}_periodic_r2', 0):.3f})"
            for j in range(d)
        ],
        horizontal_spacing=0.06,
    )

    vmin, vmax = float(chars_sub.min()), float(chars_sub.max())

    for j in range(d):
        c = j + 1

        # Scatter
        fig.add_trace(
            go.Scattergl(
                x=chars_sub,
                y=acts_sub[:, j],
                mode="markers",
                marker=dict(
                    color=chars_sub,
                    colorscale="Viridis",
                    cmin=vmin,
                    cmax=vmax,
                    size=1.0,
                    opacity=0.15,
                    showscale=(j == d - 1),
                    colorbar=dict(title="chars", len=0.8) if j == d - 1 else None,
                ),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=c,
        )

        # Linear fit
        c_lin = np.polyfit(chars_full, acts_full[:, j], 1)
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=np.polyval(c_lin, x_fit),
                mode="lines",
                line=dict(color="red", width=2.5, dash="dash"),
                name="Linear" if j == 0 else None,
                showlegend=(j == 0),
            ),
            row=1,
            col=c,
        )

        # Periodic fit
        beta, *_ = np.linalg.lstsq(X_per_full, acts_full[:, j], rcond=None)
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=X_per_fit @ beta,
                mode="lines",
                line=dict(color="blue", width=2.5),
                name="Periodic" if j == 0 else None,
                showlegend=(j == 0),
            ),
            row=1,
            col=c,
        )

        fig.update_xaxes(title_text="chars_since_nl", row=1, col=c)
        fig.update_yaxes(title_text=f"dim {j}", row=1, col=c)

    # Build summary line for title
    method_scores = "  ".join(
        f"{METHOD_SHORT[m]}={scores_row.get(m, 0):.4f}" for m in VALID_RANK_BY
    )
    fig.update_layout(
        title=f"Expert {expert_id} — Per-Dim Analysis  ({method_scores})",
        height=400,
        width=max(350 * d + 100, 800),
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    logger.info(f"Saved dim analysis → {output_path}")
    return fig


# ═══════════════════════════ CLI ═════════════════════════════════════════

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    # Model & data
    model_name: str = typer.Option("google/gemma-2-9b"),
    dataset_name: str = typer.Option("monology/pile-uncopyrighted"),
    output_path: str = typer.Option("outputs"),
    batch_size: int = typer.Option(8),
    num_workers: int = typer.Option(0),
    # Text processing
    line_length: int = typer.Option(80),
    num_samples: int = typer.Option(100),
    min_lines: int = typer.Option(5),
    max_seq_len: int = typer.Option(2048),
    use_chat: bool = typer.Option(False),
    # SMIXAE
    smixae_path: str = typer.Option(..., help="SMIXAE checkpoint dir."),
    hook_name: str = typer.Option("blocks.20.hook_resid_post"),
    sae_batch_size: int = typer.Option(4096),
    # Scoring
    n_harmonics: int = typer.Option(3, help="Fourier harmonics for periodic scoring."),
    # Visualisation
    plot_top_k: int = typer.Option(10, help="Top-k experts to plot per method."),
    plot_max_points: int = typer.Option(50_000),
    dim_analysis_top_n: int = typer.Option(
        3, help="Generate dim analysis for top N experts per method."
    ),
    # Misc
    seed: int = typer.Option(42),
) -> None:
    """Analyse SMIXAE experts for newline-position manifold structure."""

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    # Output dir
    model_slug = os.path.basename(model_name.strip("/"))
    ds_slug = os.path.basename(dataset_name.strip("/"))
    out_dir = os.path.join(output_path, model_slug, ds_slug)
    os.makedirs(out_dir, exist_ok=True)

    # ── Model via TransformerLens ────────────────────────────────────────
    dtype = (
        torch.float32
        if any(k in model_name for k in ("gpt2", "pythia"))
        else torch.bfloat16
    )
    logger.info(f"Loading {model_name} via TransformerLens  {gpu_mem_mb()}")
    model = HookedTransformer.from_pretrained(
        model_name,
        dtype=dtype,
        device=str(device),
    )
    model.eval()
    logger.info(
        f"Loaded ({model.cfg.n_layers} layers, n_ctx={model.cfg.n_ctx})  {gpu_mem_mb()}"
    )

    if hook_name not in model.hook_dict:
        resid = sorted(h for h in model.hook_dict if "resid" in h)[:20]
        raise typer.BadParameter(f"Unknown hook '{hook_name}'. Residual hooks: {resid}")
    logger.info(f"Hook: {hook_name}  (layer={extract_layer_from_hook(hook_name)})")

    tokenizer = model.tokenizer
    assert tokenizer is not None
    tokenizer.padding_side = "left" if "Qwen3" in model_name else "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert 0 < max_seq_len <= model.cfg.n_ctx

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

    dataset = dataset.map(make_line_wrapper(line_length), desc="Wrapping")
    dataset = dataset.map(
        make_forward_inputs_with_chars_since_nl(tokenizer, max_seq_len, use_chat),
        batched=True,
        desc="Tokenizing",
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
        desc="Validating",
    )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Collect hidden states at hook, then FREE the model
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"Collecting hidden states at {hook_name}  {gpu_mem_mb()}")
    hiddens_list = collect_hook_hiddens(
        dataset,
        model,
        tokenizer,
        hook_name,
        batch_size,
        num_workers,
        max_seq_len,
    )

    all_hiddens = torch.cat(hiddens_list, dim=0)
    all_labels = torch.cat(
        [torch.as_tensor(c, dtype=torch.long) for c in dataset["chars_since_nl"]]
    )
    assert all_hiddens.shape[0] == all_labels.shape[0]

    keep = all_labels > 0
    all_hiddens = all_hiddens[keep]
    all_labels = all_labels[keep]
    logger.info(f"Tokens (label>0): {all_hiddens.shape[0]:,}")

    del model, hiddens_list
    flush_gpu()
    logger.info(f"Model freed  {gpu_mem_mb()}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: SMIXAE encode → CPU, then FREE
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"Loading SMIXAE from {smixae_path}  {gpu_mem_mb()}")
    sae = SAE.load_from_disk(path=smixae_path, device=str(device))
    sae.eval()
    logger.info(
        f"SMIXAE: {sae.cfg.n_experts} experts, "
        f"d_expert={sae.cfg.d_expert}, d_bottleneck={sae.cfg.d_bottleneck}  "
        f"{gpu_mem_mb()}"
    )

    expert_acts = extract_expert_bottleneck_acts(sae, all_hiddens, sae_batch_size)
    logger.info(f"Expert activations: {expert_acts.shape}  {gpu_mem_mb()}")

    threshold = float(sae.threshold.item())
    n_experts_cfg = sae.cfg.n_experts
    d_bn_cfg = sae.cfg.d_bottleneck

    del sae, all_hiddens
    flush_gpu()
    logger.info(f"SMIXAE freed  {gpu_mem_mb()}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Scoring + Visualisation (all CPU)
    # ══════════════════════════════════════════════════════════════════════
    scores_df = compute_expert_scores(
        expert_acts,
        all_labels,
        line_length,
        n_harmonics,
    )

    # Log top experts per method
    logger.info(f"\n{'═' * 70}")
    display_cols = ["expert_id"] + VALID_RANK_BY
    for method in VALID_RANK_BY:
        ranked = scores_df.sort_values(method, ascending=False)
        logger.info(
            f"\nTop-10 by {method}:\n"
            f"{ranked[display_cols].head(10).to_string(index=False)}"
        )

    # Class stats (original fine-grained labels)
    class_means, firing_rates, class_labels_arr = compute_expert_class_stats(
        expert_acts,
        all_labels,
        threshold=threshold,
    )

    # ── Multi-method plots ───────────────────────────────────────────────
    plot_multi_method_scatter(
        expert_acts,
        all_labels,
        scores_df,
        top_k=plot_top_k,
        max_points=plot_max_points,
        output_path=os.path.join(out_dir, "top_experts_scatter.html"),
    )
    plot_multi_method_class_means(
        class_means,
        class_labels_arr,
        scores_df,
        top_k=plot_top_k,
        output_path=os.path.join(out_dir, "top_experts_class_means.html"),
    )

    # ── Dim analysis for unique top experts across all methods ────────────
    top_expert_ids: set[int] = set()
    for method in VALID_RANK_BY:
        top_expert_ids.update(
            scores_df.sort_values(method, ascending=False)
            .head(dim_analysis_top_n)["expert_id"]
            .values.tolist()
        )

    logger.info(
        f"Generating dim analysis for {len(top_expert_ids)} unique experts "
        f"(top-{dim_analysis_top_n} per method)"
    )
    for eid in sorted(top_expert_ids):
        row = scores_df[scores_df["expert_id"] == eid].iloc[0]
        plot_expert_dim_analysis(
            expert_acts,
            all_labels,
            int(eid),
            row.to_dict(),
            line_length,
            n_harmonics,
            max_points=max(plot_max_points // 3, 5000),
            output_path=os.path.join(out_dir, f"dim_analysis_expert{int(eid)}.html"),
        )

    # ── Save numerical results ───────────────────────────────────────────
    scores_df.to_csv(os.path.join(out_dir, "expert_scores.csv"), index=False)
    np.save(os.path.join(out_dir, "expert_class_means.npy"), class_means)
    np.save(os.path.join(out_dir, "expert_firing_rates.npy"), firing_rates)
    np.save(os.path.join(out_dir, "class_labels.npy"), class_labels_arr)

    summary: dict[str, Any] = {
        "hook_name": hook_name,
        "n_experts": n_experts_cfg,
        "d_bottleneck": d_bn_cfg,
        "n_harmonics": n_harmonics,
        "line_length": line_length,
        "n_tokens": int(all_labels.shape[0]),
        "n_classes": int(len(class_labels_arr)),
        "threshold": threshold,
    }
    for method in VALID_RANK_BY:
        top = scores_df.sort_values(method, ascending=False)
        summary[f"top1_{method}_expert"] = int(top.iloc[0]["expert_id"])
        summary[f"top1_{method}"] = float(top.iloc[0][method])
        summary[f"top5_mean_{method}"] = float(top.head(5)[method].mean())
        summary[f"top10_mean_{method}"] = float(top.head(10)[method].mean())

    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "summary.csv"),
        index=False,
    )
    logger.info(f"\nResults saved to {out_dir}")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    app()
