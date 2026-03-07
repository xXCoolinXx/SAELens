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
        self.expert_activations = expert_activations[active_mask].float()
        self.llm_activations = llm_activations[active_mask].float()

        assert expert_activations.shape[0] == llm_activations.shape[0], (
            "Different number of expert activations and embeddings. Did you forget to pre-filter?"
        )

        self.expert_id = expert_id
        self.active_indices = active_mask.nonzero().cpu().tolist()
        self.local_continuity_scores: torch.Tensor | None = None

    def evaluate_manifold(self, k_neighbors: int = 10) -> torch.Tensor:
        # Compute Euclidean distance in the bottleneck space
        dists = torch.cdist(self.expert_activations, self.expert_activations)

        _, indices = torch.topk(dists, k=k_neighbors + 1, largest=False)

        # Remove the point itself
        indices = indices[:, 1:]

        # Normalize to compute cosine_similarity - this should be a somewhat useful distant metric
        llm_activations = F.normalize(self.llm_activations, p=2, dim=-1)

        llm_neighbors = llm_activations[indices]

        center = llm_activations.unsqueeze(1)

        sims = (center * llm_neighbors).sum(dim=-1)
        self.local_continuity_scores = sims.mean(dim=-1)

        return self.local_continuity_scores

    def get_context_windows(
        self, str_tokens: list[list[str]], context_window: int = 10
    ) -> list[str]:
        contexts = []

        # Unpack the exact [batch, seq] coordinate for every active point
        for batch_idx, seq_idx in self.active_indices:
            seq = str_tokens[batch_idx]

            # Calculate bounds, preventing index out-of-bounds
            start = max(0, seq_idx - context_window)
            end = min(len(seq), seq_idx + context_window + 1)

            # Slice the list and bold the exact token that fired
            window = list(seq[start:end])
            target_rel_idx = seq_idx - start
            window[target_rel_idx] = f"<b>[{window[target_rel_idx]}]</b>"

            # Join back to a string and format linebreaks for Plotly
            contexts.append("".join(window).replace("\n", "<br>"))

        return contexts

    def get_plot(
        self,
        str_tokens: list[list[str]],
        k_neighbors: int = 10,
        context_window: int = 10,
    ) -> Figure:
        """
        Evaluates the manifold, builds context strings, and returns an interactive 3D plot.
        """
        # 1. Compute geometry and text contexts
        if self.local_continuity_scores is None:
            self.evaluate_manifold(k_neighbors=k_neighbors)
        contexts = self.get_context_windows(str_tokens, context_window=context_window)

        # 2. Detach and move to CPU/NumPy
        pts = self.expert_activations.detach().cpu().numpy()
        scores_np = self.local_continuity_scores.detach().cpu().numpy()  # type: ignore

        df = pd.DataFrame(
            {
                "x": pts[:, 0],
                "y": pts[:, 1],
                "z": pts[:, 2],
                "Continuity Score": scores_np,
                "Context": contexts,
            }
        )

        # 4. Generate the Plotly figure
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

        # Adjust marker size for better visibility
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

    print(
        f"Streaming {dataset_name} (Note: using already downloaded datasets is not yet implemented)"
    )
    dataset = load_dataset(dataset_name, split=None, streaming=True)

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

    # Collect batch-wise
    for i in tqdm(range(0, n_input_samples, llm_batch_size)):
        batch = tokenized[i : i + llm_batch_size, :].to(device)

        # Run the batch through the model

        _, cache = model.run_with_cache(batch, names_filter=hook_name)
        acts = cache[hook_name]

        all_acts.append(acts)

    activations = torch.cat(all_acts, dim=0).to(device)

    str_tokens = model.to_str_tokens(tokenized)

    return activations, str_tokens


def get_sae_activations(
    checkpoint_path: str,
    device: str,
    activations: torch.Tensor,
    sae_batch_size: int,
    active_threshold: float = 1e-5,
    min_points: int = 15,
) -> list[Expert]:
    print(f"Loading SAE from {checkpoint_path}")
    sae = SAE.load_from_disk(path=checkpoint_path, device=device)

    sae_activations = []

    with torch.no_grad():
        # z = sae.encode(activations)

        for i in tqdm(range(0, len(activations), sae_batch_size)):
            batch = activations[i : i + sae_batch_size, ...]

            _, cache = sae.run_with_cache(
                batch, names_filter=["hook_sae_acts_post", "hook_decode_mask"]
            )

            active_experts = cache["hook_sae_acts_post"] * cache[
                "hook_decode_mask"
            ].unsqueeze(-1)

            sae_activations.append(active_experts)

    sae_activations = torch.cat(sae_activations, dim=0).to(device)

    # Package experts
    experts = []
    n_experts = sae_activations.shape[-2]  # Get n_experts

    for i in tqdm(range(n_experts)):
        # Grab points
        expert_pts = sae_activations[..., i, :]

        active_mask = torch.norm(expert_pts, p=2, dim=-1) > active_threshold

        # Check if there are enough points to do anything interesting
        if active_mask.sum() >= min_points:
            expert = Expert(
                active_mask,
                i,
                activations,
                expert_pts,
            )

            experts.append(expert)

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
    # Parse the checkpoint path to dynamically create the output subfolder
    # Example: /scratch/.../qkkhpjf1/final_250003456 -> folder: qkkhpjf1_250003456
    ckpt_path_obj = Path(checkpoint_path)
    run_hash = ckpt_path_obj.parent.name
    run_step = ckpt_path_obj.name.replace("final_", "")

    run_folder_name = f"{run_hash}_{run_step}"
    final_output_dir = os.path.join(output_dir, run_folder_name)

    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Plots will be saved to: {final_output_dir}")

    # 1. Collect Base LLM Activations
    llm_acts, str_tokens = collect_activations(
        model_name=base_model_name,
        hook_name=hook_point,
        dataset_name=dataset_name,
        max_length=input_sequence_length,
        n_input_samples=n_input_samples,
        device=device,
        llm_batch_size=llm_batch_size,
    )

    # 2. Extract and Package SAE Experts
    experts = get_sae_activations(
        checkpoint_path=checkpoint_path,
        device=device,
        activations=llm_acts,
        sae_batch_size=sae_batch_size,
        active_threshold=active_threshold,
        min_points=min_points,
    )

    if not experts:
        print("No experts fired enough times to exceed the min_points threshold.")
        return

    # 3. Compute geometric continuity
    print(f"Evaluating manifolds for continuity (k={k_neighbors})...")
    for expert in tqdm(experts):
        expert.evaluate_manifold(k_neighbors=k_neighbors)

    # 4. Sort descending by the mean continuity score
    experts.sort(key=lambda e: e.local_continuity_scores.mean().item(), reverse=True)  # type: ignore

    # 5. Output and Plot
    print(
        f"\nSaving top {n_interesting_experts_to_plot} most structured experts to '{final_output_dir}/'..."
    )
    for i in range(min(n_interesting_experts_to_plot, len(experts))):
        expert = experts[i]
        mean_score = expert.local_continuity_scores.mean().item()  # type: ignore

        print(
            f"Rank {i + 1:02d} | Expert {expert.expert_id:4d} | Mean Continuity: {mean_score:.4f} | Points: {expert.expert_activations.shape[0]}"
        )

        fig = expert.get_plot(
            str_tokens=str_tokens,  # type: ignore
            k_neighbors=k_neighbors,
            context_window=context_window_display,
        )

        # Save to HTML with a clean, sortable naming scheme
        filename = f"rank_{i + 1:02d}_expert_{expert.expert_id:04d}_score_{mean_score:.4f}.html"
        filepath = os.path.join(final_output_dir, filename)

        fig.write_html(filepath)


if __name__ == "__main__":
    typer.run(main)


# python analyze_experts.py \
#     --checkpoint-path "/scratch/Collin/SAELens/checkpoints/qkkhpjf1/final_250003456" \
#     --nase-model-name "your-model-here" \
#     --hook-point "your-hook-point" \
#     --n-input-samples 1500 \
#     --input-sequence-length 128 \
#     --n-interesting-experts-to-plot 100 \
#     --output-dir "manifold_analysis" \
#     --k-neighbors 10 \
#     --device "cuda"
