import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
import typer
from datasets import load_dataset
from plotly.graph_objects import Figure
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE


class Expert:
    def __init__(
        self,
        active_mask: torch.Tensor,
        expert_id: int,
        llm_activations: torch.Tensor,
        expert_activations: torch.Tensor,
        labels: torch.Tensor | None = None,
        seq_positions: torch.Tensor | None = None,
    ):
        """
        Parameters
        ----------
        seq_positions : (B,) tensor, optional
            Maps each batch index to the *original* sequence position of the
            token that was encoded.  Only set in last-token-only mode so that
            context windows render around the correct position.
        """
        self.expert_activations = expert_activations[active_mask].float().cpu()
        self.llm_activations = llm_activations[active_mask].float().cpu()
        self.labels = labels[active_mask].long().cpu() if labels is not None else None
        self.seq_positions = seq_positions  # kept as full (B,) lookup table

        assert self.expert_activations.shape[0] == self.llm_activations.shape[0], (
            "Different number of expert activations and embeddings. "
            "Did you forget to pre-filter?"
        )

        self.expert_id = expert_id
        self.active_indices = active_mask.nonzero().cpu().tolist()
        self.local_continuity_scores: torch.Tensor | None = None
        self.concept_purity_scores: torch.Tensor | None = None
        self.concept_scores: dict[str, torch.Tensor] = {}

    # ── unified accessor ──────────────────────────────────────────────
    @property
    def scores(self) -> torch.Tensor:
        if self.concept_purity_scores is not None:
            return self.concept_purity_scores
        if self.local_continuity_scores is not None:
            return self.local_continuity_scores
        raise ValueError(
            "No scores computed yet. Call evaluate_manifold or evaluate_concept first."
        )

    @property
    def score_name(self) -> str:
        if self.concept_purity_scores is not None:
            return "Purity"
        return "Continuity"

    # ── shared KNN helper ─────────────────────────────────────────────
    def _evaluate_util(
        self, k_neighbors: int = 10, device: str = "cuda"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expert_acts_gpu = self.expert_activations.to(device)
        llm_acts_gpu = self.llm_activations.to(device)

        dists = torch.cdist(expert_acts_gpu, expert_acts_gpu)
        _, indices = torch.topk(dists, k=k_neighbors + 1, largest=False)
        indices = indices[:, 1:]
        del dists

        return llm_acts_gpu, expert_acts_gpu, indices

    # ── unlabelled metric ─────────────────────────────────────────────
    def evaluate_manifold(
        self, k_neighbors: int = 10, device: str = "cuda"
    ) -> torch.Tensor:
        llm_acts_gpu, expert_acts_gpu, indices = self._evaluate_util(
            k_neighbors, device
        )

        llm_normed = F.normalize(llm_acts_gpu, p=2, dim=-1)
        llm_neighbors = llm_normed[indices]
        center = llm_normed.unsqueeze(1)
        sims = (center * llm_neighbors).sum(dim=-1)

        self.local_continuity_scores = sims.mean(dim=-1).cpu()

        del expert_acts_gpu, llm_acts_gpu, llm_normed, llm_neighbors, center, sims
        torch.cuda.empty_cache()

        return self.local_continuity_scores

    # ── labelled metric ───────────────────────────────────────────────
    def evaluate_concept(
        self,
        concept: str | None = None,
        concept_labels: torch.Tensor | None = None,
        k_neighbors: int = 10,
        device: str = "cuda",
    ) -> torch.Tensor:
        _, _, indices = self._evaluate_util(k_neighbors, device)
        indices = indices.cpu()
        torch.cuda.empty_cache()

        if concept_labels is not None:
            labels = concept_labels.long()
        elif self.labels is not None:
            labels = self.labels
        else:
            raise ValueError(
                "No labels provided to evaluate_concept and none stored on Expert."
            )

        neighbor_labels = labels[indices]
        center_labels = labels.unsqueeze(1).expand_as(neighbor_labels)
        purity = (neighbor_labels == center_labels).float().mean(dim=-1)

        if concept is not None:
            self.concept_scores[concept] = purity
        self.concept_purity_scores = purity

        return purity

    # ── context windows ───────────────────────────────────────────────
    def get_context_windows(
        self, str_tokens: list[list[str]], context_window: int = 10
    ) -> list[str]:
        contexts = []
        for batch_idx, seq_idx in self.active_indices:
            seq = str_tokens[batch_idx]

            # In last-token-only mode the internal seq_idx is always 0;
            # map it back to the real position in the full sequence.
            if self.seq_positions is not None:
                seq_idx = int(self.seq_positions[batch_idx].item())

            start = max(0, seq_idx - context_window)
            end = min(len(seq), seq_idx + context_window + 1)
            window = list(seq[start:end])
            target_rel_idx = seq_idx - start
            window[target_rel_idx] = f"<b>[{window[target_rel_idx]}]</b>"
            contexts.append("".join(window).replace("\n", "<br>"))
        return contexts

    # ── plotting ──────────────────────────────────────────────────────
    def get_plot(
        self,
        str_tokens: list[list[str]],
        k_neighbors: int = 10,
        context_window: int = 10,
        device: str = "cuda",
        label_names: dict[int, str] | None = None,
    ) -> Figure:
        if self.concept_purity_scores is None and self.local_continuity_scores is None:
            if self.labels is not None:
                self.evaluate_concept(k_neighbors=k_neighbors, device=device)
            else:
                self.evaluate_manifold(k_neighbors=k_neighbors, device=device)

        contexts = self.get_context_windows(str_tokens, context_window=context_window)
        pts = self.expert_activations.numpy()
        scores_np = self.scores.numpy()
        mean_score = float(scores_np.mean())
        score_col = f"{self.score_name} Score"

        use_label_colour = self.labels is not None and label_names is not None

        if use_label_colour:
            label_strs = [
                label_names.get(l.item(), str(l.item()))
                for l in self.labels  # type: ignore[union-attr]
            ]
            df = pd.DataFrame(
                {
                    "x": pts[:, 0],
                    "y": pts[:, 1],
                    "z": pts[:, 2],
                    score_col: scores_np,
                    "Label": label_strs,
                    "Context": contexts,
                }
            )
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                color="Label",
                hover_data={
                    "Context": True,
                    score_col: True,
                    "x": False,
                    "y": False,
                    "z": False,
                },
                title=(
                    f"Expert {self.expert_id}  "
                    f"(n={pts.shape[0]}, mean {self.score_name.lower()}={mean_score:.4f})"
                ),
                opacity=0.8,
            )
        else:
            df = pd.DataFrame(
                {
                    "x": pts[:, 0],
                    "y": pts[:, 1],
                    "z": pts[:, 2],
                    score_col: scores_np,
                    "Context": contexts,
                }
            )
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                color=score_col,
                color_continuous_scale="Viridis",
                hover_data={"Context": True, "x": False, "y": False, "z": False},
                title=(
                    f"Expert {self.expert_id}  "
                    f"(n={pts.shape[0]}, mean {self.score_name.lower()}={mean_score:.4f})"
                ),
                opacity=0.8,
            )

        fig.update_traces(marker=dict(size=4))
        return fig


# ======================================================================
# Data loading + LLM activation collection
# ======================================================================
def collect_activations(
    model_name: str,
    hook_name: str,
    max_length: int,
    n_input_samples: int,
    device: str,
    llm_batch_size: int,
    dataset_name: str | None = None,
    dataframe_path: str | None = None,
    text_column: str = "text",
    label_column: str | None = None,
) -> tuple[
    torch.Tensor,
    list[list[str]],
    torch.Tensor | None,
    dict[int, str] | None,
    torch.Tensor,
]:
    """
    Returns
    -------
    activations          : (B, S, D) on CPU
    str_tokens           : list[list[str]] — full sequences for context windows
    labels               : (B, S) long tensor on CPU, or None
    label_names          : {int → str} mapping, or None
    last_token_positions : (B,) long tensor — index of last non-pad token per seq
    """
    print(f"Loading {model_name} to collect activations")
    model = HookedTransformer.from_pretrained(model_name, device=device)

    # ── 1. Load texts (+ optional raw labels) ────────────────────────
    texts: list[str] = []
    raw_labels: list[str | int] | None = [] if label_column else None

    if dataframe_path is not None:
        print(f"Loading data from {dataframe_path}")
        ext = Path(dataframe_path).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(dataframe_path)
        elif ext in (".parquet", ".pq"):
            df = pd.read_parquet(dataframe_path)
        elif ext in (".json", ".jsonl"):
            df = pd.read_json(dataframe_path, lines=(ext == ".jsonl"))
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        texts = df[text_column].tolist()[:n_input_samples]
        if raw_labels is not None:
            raw_labels = df[label_column].tolist()[:n_input_samples]  # type: ignore[index]

    elif dataset_name is not None:
        print(f"Streaming {dataset_name}")
        dataset = load_dataset(dataset_name, streaming=True, split="train")
        for i, sample in enumerate(dataset):
            if i == n_input_samples:
                break
            texts.append(sample[text_column])
            if raw_labels is not None:
                raw_labels.append(sample[label_column])  # type: ignore[index]
    else:
        raise ValueError("Provide either --dataset-name or --dataframe-path.")

    # ── 2. Encode labels → integer ids ────────────────────────────────
    labels_tensor: torch.Tensor | None = None
    label_names: dict[int, str] | None = None

    if raw_labels is not None and len(raw_labels) > 0:
        unique = sorted({str(l) for l in raw_labels})
        label_to_id = {l: i for i, l in enumerate(unique)}
        label_names = {i: l for l, i in label_to_id.items()}
        label_ids = torch.tensor(
            [label_to_id[str(l)] for l in raw_labels], dtype=torch.long
        )
        print(
            f"Found {len(unique)} unique labels: "
            f"{', '.join(unique[:10])}{'…' if len(unique) > 10 else ''}"
        )
    else:
        label_ids = None

    # ── 3. Tokenise ───────────────────────────────────────────────────
    print("Collecting activations...")
    tokenized = model.to_tokens(texts, prepend_bos=True, move_to_device=False)[
        :, :max_length
    ]
    B, S = tokenized.shape

    # Broadcast document-level labels → token-level (B, S)
    if label_ids is not None:
        labels_tensor = label_ids[:B].unsqueeze(1).expand(B, S).clone()

    # ── 4. Find last non-pad position per sequence ────────────────────
    pad_token_id = getattr(model.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        non_pad_mask = tokenized != pad_token_id  # (B, S)
        # Build column indices, mask out pad positions with -1, then take max
        col_indices = torch.arange(S).unsqueeze(0).expand(B, S)
        last_token_positions = (
            col_indices.masked_fill(~non_pad_mask, -1).max(dim=1).values
        )
        last_token_positions = last_token_positions.clamp(min=0)
    else:
        last_token_positions = torch.full((B,), S - 1, dtype=torch.long)

    print(
        f"Last non-pad positions — min: {last_token_positions.min().item()}, "
        f"max: {last_token_positions.max().item()}, "
        f"mean: {last_token_positions.float().mean().item():.1f} "
        f"(seq length {S})"
    )

    # ── 5. Forward pass ───────────────────────────────────────────────
    all_acts = []
    for i in tqdm(range(0, B, llm_batch_size)):
        batch = tokenized[i : i + llm_batch_size].to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=hook_name)
        all_acts.append(cache[hook_name].cpu())
        del cache
        torch.cuda.empty_cache()

    activations = torch.cat(all_acts, dim=0)
    del all_acts

    str_tokens = [model.to_str_tokens(tokenized[i]) for i in range(B)]

    del model
    torch.cuda.empty_cache()
    print("LLM deleted and GPU memory freed.")

    return activations, str_tokens, labels_tensor, label_names, last_token_positions


# ======================================================================
# SAE encoding → Expert construction
# ======================================================================
def get_sae_activations(
    checkpoint_path: str,
    device: str,
    activations: torch.Tensor,
    sae_batch_size: int,
    active_threshold: float = 1e-5,
    min_points: int = 15,
    max_points: int = 1000,
    labels: torch.Tensor | None = None,
    last_token_only: bool = False,
    last_token_positions: torch.Tensor | None = None,
) -> list[Expert]:
    print(f"Loading SAE from {checkpoint_path}")
    sae = SAE.load_from_disk(path=checkpoint_path, device=device)

    B, S_full, D = activations.shape

    # ── When labelled, only encode the last *non-pad* token per seq ───
    seq_positions: torch.Tensor | None = None
    if last_token_only:
        if last_token_positions is None:
            raise ValueError(
                "last_token_only=True but no last_token_positions provided."
            )

        seq_positions = last_token_positions  # (B,) — lookup for context windows

        # Gather the single token per sequence
        # activations: (B, S_full, D) → (B, 1, D)
        gather_idx = last_token_positions.unsqueeze(1).unsqueeze(2).expand(B, 1, D)
        activations = activations.gather(1, gather_idx)

        if labels is not None:
            label_idx = last_token_positions.unsqueeze(1)  # (B, 1)
            labels = labels.gather(1, label_idx)

        S = 1
        print(
            f"last_token_only=True → gathered 1 token per sequence "
            f"({B * S_full} → {B} tokens through SAE)"
        )
    else:
        S = S_full

    activations_flat = activations.reshape(B * S, D)

    sae_activations: list[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, B * S, sae_batch_size)):
            batch = activations_flat[i : i + sae_batch_size].to(device)
            _, cache = sae.run_with_cache(
                batch, names_filter=["hook_sae_acts_post", "hook_decode_mask"]
            )
            active_experts = cache["hook_sae_acts_post"] * cache[
                "hook_decode_mask"
            ].unsqueeze(-1)
            sae_activations.append(active_experts.cpu())
            del cache, active_experts, batch
            torch.cuda.empty_cache()

    del sae
    torch.cuda.empty_cache()
    print("SAE deleted and GPU memory freed.")

    sae_activations_cat = torch.cat(sae_activations, dim=0)
    del sae_activations

    sae_activations_cat = sae_activations_cat.view(
        B, S, sae_activations_cat.shape[1], sae_activations_cat.shape[2]
    )

    experts: list[Expert] = []
    n_experts = sae_activations_cat.shape[-2]

    for i in tqdm(range(n_experts)):
        expert_pts = sae_activations_cat[..., i, :]  # (B, S, D')
        active_mask = torch.norm(expert_pts, p=2, dim=-1) > active_threshold
        n_active = int(active_mask.sum().item())

        if n_active < min_points:
            continue

        if max_points and n_active > max_points:
            active_indices = active_mask.nonzero(as_tuple=False)
            perm = torch.randperm(n_active)[:max_points]
            chosen = active_indices[perm]
            sampled_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            sampled_mask[chosen[:, 0], chosen[:, 1]] = True
            active_mask = sampled_mask

        expert = Expert(
            active_mask,
            i,
            activations,
            expert_pts,
            labels=labels,
            seq_positions=seq_positions,
        )
        experts.append(expert)

    del sae_activations_cat
    return experts


# ======================================================================
# CLI entry-point
# ======================================================================
def main(
    checkpoint_path: str = typer.Option(..., help="Path to your checkpoint"),
    base_model_name: str = typer.Option(..., help="Model to load"),
    hook_point: str = typer.Option(
        ..., help="The hook point where your SAE was trained"
    ),
    # ── data source (supply one) ──
    dataset_name: str | None = typer.Option(
        None, help="HF dataset to stream (unlabelled)"
    ),
    dataframe_path: str | None = typer.Option(
        None, help="Path to a CSV / Parquet / JSON(L) file"
    ),
    text_column: str = typer.Option("text", help="Column containing text"),
    label_column: str | None = typer.Option(
        None,
        help="Column containing labels (enables concept-purity evaluation, "
        "last-token-only mode)",
    ),
    # ── general ──
    n_input_samples: int = typer.Option(1000, help="Number of input texts to sample"),
    input_sequence_length: int = typer.Option(128, help="Input sequence length"),
    context_window_display: int = typer.Option(
        10, help="Number of surrounding tokens to display"
    ),
    n_interesting_experts_to_plot: int = typer.Option(
        50, help="Number of top experts to plot"
    ),
    device: str = typer.Option("cuda", help="Device to load models on"),
    llm_batch_size: int = typer.Option(16, help="Batch size for base LLM inference"),
    sae_batch_size: int = typer.Option(2048, help="Batch size for SAE inference"),
    k_neighbors: int = typer.Option(10, help="Number of neighbors for KNN evaluation"),
    active_threshold: float = typer.Option(
        1e-5, help="L2 norm threshold to consider an expert active"
    ),
    min_points: int = typer.Option(
        15, help="Minimum active tokens required to evaluate an expert"
    ),
    max_points: int = typer.Option(
        1000, help="Max active tokens per expert (0 = no cap)"
    ),
    output_dir: str = typer.Option(
        "expert_plots", help="Base directory to save the HTML plots"
    ),
):
    if dataset_name is None and dataframe_path is None:
        dataset_name = "monology/pile-uncopyrighted"
        print(f"No data source specified — defaulting to {dataset_name}")

    ckpt_path_obj = Path(checkpoint_path)
    run_hash = ckpt_path_obj.parent.name
    run_step = ckpt_path_obj.name.replace("final_", "")
    run_folder_name = f"{run_hash}_{run_step}"
    final_output_dir = os.path.join(output_dir, run_folder_name)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Plots will be saved to: {final_output_dir}")

    # ── 1. Collect LLM activations (+ optional labels) ────────────────
    llm_acts, str_tokens, labels, label_names, last_token_positions = (
        collect_activations(
            model_name=base_model_name,
            hook_name=hook_point,
            max_length=input_sequence_length,
            n_input_samples=n_input_samples,
            device=device,
            llm_batch_size=llm_batch_size,
            dataset_name=dataset_name,
            dataframe_path=dataframe_path,
            text_column=text_column,
            label_column=label_column,
        )
    )

    is_labelled = labels is not None

    # ── 2. SAE encoding ───────────────────────────────────────────────
    experts = get_sae_activations(
        checkpoint_path=checkpoint_path,
        device=device,
        activations=llm_acts,
        sae_batch_size=sae_batch_size,
        active_threshold=active_threshold,
        min_points=min_points,
        max_points=max_points,
        labels=labels,
        last_token_only=is_labelled,
        last_token_positions=last_token_positions,
    )

    del llm_acts, labels

    if not experts:
        print("No experts fired enough times to exceed the min_points threshold.")
        return

    # ── 3. Evaluate ───────────────────────────────────────────────────
    if is_labelled:
        metric_name = "Purity"
        print(f"Evaluating concept purity (k={k_neighbors})…")
        for expert in tqdm(experts):
            expert.evaluate_concept(
                concept=label_column, k_neighbors=k_neighbors, device=device
            )
    else:
        metric_name = "Continuity"
        print(f"Evaluating manifold continuity (k={k_neighbors})…")
        for expert in tqdm(experts):
            expert.evaluate_manifold(k_neighbors=k_neighbors, device=device)

    # ── 4. Sort (best first) ──────────────────────────────────────────
    experts.sort(key=lambda e: e.scores.mean().item(), reverse=True)

    # ── 5. Plot ───────────────────────────────────────────────────────
    n_to_plot = min(n_interesting_experts_to_plot, len(experts))
    print(f"\nSaving top {n_to_plot} experts to '{final_output_dir}/'…")

    for i in range(n_to_plot):
        expert = experts[i]
        mean_score = expert.scores.mean().item()

        print(
            f"Rank {i + 1:02d} | Expert {expert.expert_id:4d} | "
            f"Mean {metric_name}: {mean_score:.4f} | "
            f"Points: {expert.expert_activations.shape[0]}"
        )

        fig = expert.get_plot(
            str_tokens=str_tokens,  # type: ignore[arg-type]
            k_neighbors=k_neighbors,
            context_window=context_window_display,
            device=device,
            label_names=label_names,
        )

        filename = (
            f"rank_{i + 1:02d}_expert_{expert.expert_id:04d}"
            f"_{metric_name.lower()}_{mean_score:.4f}.html"
        )
        fig.write_html(os.path.join(final_output_dir, filename))


if __name__ == "__main__":
    typer.run(main)
