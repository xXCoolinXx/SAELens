import torch
import typer
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE


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
):
    print(f"Loading SAE from {checkpoint_path}")
    sae = SAE.load_from_disk(path=checkpoint_path, device=device)

    with torch.no_grad():
        # z = sae.encode(activations)

        _, cache = sae.run_with_cache(
            activations, names_filter=["hook_sae_acts_post", "hook_decode_mask"]
        )

        active_experts = cache["hook_sae_acts_post"] * cache[
            "hook_decode_mask"
        ].unsqueeze(-1)

    return active_experts


def persistent_homology():
    pass


def geometric_categorization():
    pass


def knn_variance_metric():
    pass


def plot_experts():
    pass


def main(
    checkpoint_path: str = typer.Option(..., "Path to your checkpoint"),
    nase_model_name: str = typer.Option(..., "Model to load"),
    hook_point: str = typer.Option(..., "The hook point where your SAE was trained"),
    n_input_samples: int = typer.Option(
        1000, "Number of input texts to sample to determine manifold structure"
    ),
    context_window_display: str = typer.Option(
        10, "Number of surrounding tokens to display for each data point"
    ),
    device: str = typer.Option("cuda", "Device to load models on"),
    input_sequence_length: int = typer.Option(
        128, "Input seqeuence length. Defaults to 128 to be quick"
    ),
    n_interesting_experts_to_plot: int = typer.Option(
        50,
        "Number of interesting experts to plot - ie meaningful structure in some sense",
    ),
    dataset_name: str = typer.Option("monology/pile-uncopyrighted"),
):
    pass


if __name__ == "__main__":
    main()
