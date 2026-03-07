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
    ):
        # ── CHANGE: .cpu() so expert data lives in system RAM, not VRAM ──
        self.expert_activations = expert_activations[active_mask].float().cpu()
        self.llm_activations = llm_activations[active_mask].float().cpu()

        assert self.expert_activations.shape[0] == self.llm_activations.shape[0], (
            "Different number of expert activations and embeddings. Did you forget to pre-filter?"
        )

        self.expert_id = expert_id
        self.active_indices = active_mask.nonzero().cpu().tolist()
        self.local_continuity_scores: torch.Tensor | None = None

    # ── CHANGE: accept a device arg, move data onto GPU only for the computation ──
    def evaluate_manifold(
        self, k_neighbors: int = 10, device: str = "cuda"
    ) -> torch.Tensor:
        # Move to GPU just for this computation
        expert_acts_gpu = self.expert_activations.to(device)
        llm_acts_gpu = self.llm_activations.to(device)

        # Compute Euclidean distance in the bottleneck space
        dists = torch.cdist(expert_acts_gpu, expert_acts_gpu)

        _, indices = torch.topk(dists, k=k_neighbors + 1, largest=False)
        indices = indices[:, 1:]

        del dists  # ── CHANGE: free the big N×N matrix immediately ──

        llm_normed = F.normalize(llm_acts_gpu, p=2, dim=-1)
        llm_neighbors = llm_normed[indices]
        center = llm_normed.unsqueeze(1)
        sims = (center * llm_neighbors).sum(dim=-1)

        # ── CHANGE: result back to CPU ──
        self.local_continuity_scores = sims.mean(dim=-1).cpu()

        # ── CHANGE: free transient GPU tensors ──
        del expert_acts_gpu, llm_acts_gpu, llm_normed, llm_neighbors, center, sims
        torch.cuda.empty_cache()

        return self.local_continuity_scores

    def get_context_windows(
        self, str_tokens: list[list[str]], context_window: int = 10
    ) -> list[str]:
        contexts = []
        for batch_idx, seq_idx in self.active_indices:
            seq = str_tokens[batch_idx]
            start = max(0, seq_idx - context_window)
            end = min(len(seq), seq_idx + context_window + 1)
            window = list(seq[start:end])
            target_rel_idx = seq_idx - start
            window[target_rel_idx] = f"<b>[{window[target_rel_idx]}]</b>"
            contexts.append("".join(window).replace("\n", "<br>"))
        return contexts

    def get_plot(
        self,
        str_tokens: list[list[str]],
        k_neighbors: int = 10,
        context_window: int = 10,
        device: str = "cuda",  # ── CHANGE: pass device through ──
    ) -> Figure:
        if self.local_continuity_scores is None:
            self.evaluate_manifold(k_neighbors=k_neighbors, device=device)
        contexts = self.get_context_windows(str_tokens, context_window=context_window)

        pts = self.expert_activations.numpy()  # already on CPU
        scores_np = self.local_continuity_scores.numpy()  # type: ignore  # already on CPU

        df = pd.DataFrame(
            {
                "x": pts[:, 0],
                "y": pts[:, 1],
                "z": pts[:, 2],
                "Continuity Score": scores_np,
                "Context": contexts,
            }
        )

        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="Continuity Score",
            color_continuous_scale="Viridis",
            hover_data={"Context": True, "x": False, "y": False, "z": False},
            title=f"Expert {self.expert_id} Manifold (n={pts.shape[0]} points)",
            opacity=0.8,
        )
        fig.update_traces(marker=dict(size=4))
        return fig


def collect_activations(
    model_name: str,
    hook_name: str,
    dataset_name: str,
    max_length: int,
    n_input_samples: int,
    device: str,
    llm_batch_size: int,
) -> tuple[torch.Tensor, list[str] | list[list[str]]]:
    print(f"Loading {model_name} to collect activations")
    model = HookedTransformer.from_pretrained(model_name, device=device)

    print(f"Streaming {dataset_name}")
    dataset = load_dataset(dataset_name, streaming=True, split="train")

    print("Collecting activations...")

    texts = []
    for i, sample in enumerate(dataset):
        if i == n_input_samples:
            break
        texts.append(sample["text"])

    tokenized = model.to_tokens(texts, prepend_bos=True, move_to_device=False)[
        :, :max_length
    ]

    all_acts = []
    for i in tqdm(range(0, n_input_samples, llm_batch_size)):
        batch = tokenized[i : i + llm_batch_size, :].to(device)

        with torch.no_grad():  # ── CHANGE: no_grad for inference ──
            _, cache = model.run_with_cache(batch, names_filter=hook_name)

        # ── CHANGE: move each batch's activations to CPU immediately ──
        all_acts.append(cache[hook_name].cpu())
        del cache
        torch.cuda.empty_cache()

    # ── CHANGE: concatenate on CPU — this is your "cold storage" tensor ──
    activations = torch.cat(all_acts, dim=0)
    del all_acts

    str_tokens = [model.to_str_tokens(tokenized[i]) for i in range(tokenized.shape[0])]

    # ── CHANGE: free the entire LLM before returning ──
    del model
    torch.cuda.empty_cache()
    print("LLM deleted and GPU memory freed.")

    return activations, str_tokens


def get_sae_activations(
    checkpoint_path: str,
    device: str,
    activations: torch.Tensor,  # ── now lives on CPU ──
    sae_batch_size: int,
    active_threshold: float = 1e-5,
    min_points: int = 15,
    max_points: int = 1000,
) -> list[Expert]:
    print(f"Loading SAE from {checkpoint_path}")
    sae = SAE.load_from_disk(path=checkpoint_path, device=device)

    sae_activations = []

    # Flatten to avoid issues with SMIXAE implentation for extra seq_len dimension
    B, S, D = activations.shape
    activations_flat = activations.reshape(B * S, D)

    with torch.no_grad():
        for i in tqdm(range(0, B * S, sae_batch_size)):
            # ── CHANGE: move each batch to GPU, then result back to CPU ──
            batch = activations_flat[i : i + sae_batch_size].to(device)

            _, cache = sae.run_with_cache(
                batch, names_filter=["hook_sae_acts_post", "hook_decode_mask"]
            )

            active_experts = cache["hook_sae_acts_post"] * cache[
                "hook_decode_mask"
            ].unsqueeze(-1)

            # ── CHANGE: immediately move to CPU ──
            sae_activations.append(active_experts.cpu())
            del cache, active_experts, batch
            torch.cuda.empty_cache()

    # ── CHANGE: free the SAE ──
    del sae
    torch.cuda.empty_cache()
    print("SAE deleted and GPU memory freed.")

    # ── CHANGE: concatenate on CPU ──
    sae_activations_cat = torch.cat(sae_activations, dim=0)
    del sae_activations

    # Reshape back to original size
    sae_activations_cat = sae_activations_cat.view(
        B, S, sae_activations_cat.shape[1], sae_activations_cat.shape[2]
    )

    # Package experts — everything is on CPU now
    experts = []
    n_experts = sae_activations_cat.shape[-2]

    for i in tqdm(range(n_experts)):
        expert_pts = sae_activations_cat[..., i, :]
        # shape: (B, S, D')
        active_mask = torch.norm(expert_pts, p=2, dim=-1) > active_threshold
        # shape: (B, S)
        n_active = active_mask.sum().item()

        if n_active < min_points:
            continue

        # ── NEW: randomly subsample active points if above max_points ──
        # If max_points == 0, treat as "no cap"
        if max_points and n_active > max_points:
            # Indices of all active points in (B, S)
            active_indices = active_mask.nonzero(as_tuple=False)  # (n_active, 2)

            # Randomly choose max_points indices without replacement
            perm = torch.randperm(n_active)[:max_points]
            chosen = active_indices[perm]

            # Build a new mask that is True only at the chosen points
            sampled_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            sampled_mask[chosen[:, 0], chosen[:, 1]] = True
            active_mask = sampled_mask

        expert = Expert(active_mask, i, activations, expert_pts)
        experts.append(expert)

    # ── CHANGE: free the big concatenated tensor once experts are built ──
    del sae_activations_cat
    return experts


def main(
    checkpoint_path: str = typer.Option(..., help="Path to your checkpoint"),
    base_model_name: str = typer.Option(..., help="Model to load"),
    hook_point: str = typer.Option(
        ..., help="The hook point where your SAE was trained"
    ),
    dataset_name: str = typer.Option(
        "monology/pile-uncopyrighted", help="Dataset to stream"
    ),
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
    k_neighbors: int = typer.Option(
        10, help="Number of neighbors for local continuity (KNN)"
    ),
    active_threshold: float = typer.Option(
        1e-5, help="L2 norm threshold to consider an expert active"
    ),
    min_points: int = typer.Option(
        15, help="Minimum active tokens required to evaluate an expert"
    ),
    output_dir: str = typer.Option(
        "expert_plots", help="Base directory to save the HTML plots"
    ),
):
    ckpt_path_obj = Path(checkpoint_path)
    run_hash = ckpt_path_obj.parent.name
    run_step = ckpt_path_obj.name.replace("final_", "")
    run_folder_name = f"{run_hash}_{run_step}"
    final_output_dir = os.path.join(output_dir, run_folder_name)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Plots will be saved to: {final_output_dir}")

    # 1. Collect — LLM is freed inside this function
    llm_acts, str_tokens = collect_activations(
        model_name=base_model_name,
        hook_name=hook_point,
        dataset_name=dataset_name,
        max_length=input_sequence_length,
        n_input_samples=n_input_samples,
        device=device,
        llm_batch_size=llm_batch_size,
    )
    # At this point: GPU is empty, llm_acts is on CPU

    # 2. SAE encoding — SAE is freed inside this function
    experts = get_sae_activations(
        checkpoint_path=checkpoint_path,
        device=device,
        activations=llm_acts,
        sae_batch_size=sae_batch_size,
        active_threshold=active_threshold,
        min_points=min_points,
    )
    # At this point: GPU is empty, expert data is on CPU

    # ── CHANGE: llm_acts is no longer needed (experts have their own copies) ──
    del llm_acts

    if not experts:
        print("No experts fired enough times to exceed the min_points threshold.")
        return

    # 3. Evaluate manifolds — each expert temporarily uses GPU then frees it
    print(f"Evaluating manifolds for continuity (k={k_neighbors})...")
    for expert in tqdm(experts):
        expert.evaluate_manifold(k_neighbors=k_neighbors, device=device)

    # 4. Sort
    experts.sort(
        key=lambda e: e.local_continuity_scores.mean().item(),  # type: ignore
        reverse=True,
    )

    # 5. Plot
    print(
        f"\nSaving top {n_interesting_experts_to_plot} most structured experts to '{final_output_dir}/'..."
    )
    for i in range(min(n_interesting_experts_to_plot, len(experts))):
        expert = experts[i]
        mean_score = expert.local_continuity_scores.mean().item()  # type: ignore

        print(
            f"Rank {i + 1:02d} | Expert {expert.expert_id:4d} | "
            f"Mean Continuity: {mean_score:.4f} | Points: {expert.expert_activations.shape[0]}"
        )

        fig = expert.get_plot(
            str_tokens=str_tokens,  # type: ignore
            k_neighbors=k_neighbors,
            context_window=context_window_display,
            device=device,
        )

        filename = f"rank_{i + 1:02d}_expert_{expert.expert_id:04d}_score_{mean_score:.4f}.html"
        fig.write_html(os.path.join(final_output_dir, filename))


if __name__ == "__main__":
    typer.run(main)
