from pathlib import Path

from datasets import load_dataset

from sae_lens import (
    ContextTrainingSAEConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
)

dataset_path = "./pile_long_context"
source_repo = "monology/pile-uncopyrighted"
min_chars = 7000  # Rough proxy for 2048 tokens (approx 3.5 chars/token)

if Path(dataset_path).exists():
    print(f"Dataset already exists at {dataset_path}")  # noqa: T201
else:
    print(f"{dataset_path} not found. Streaming and filtering...")  # noqa: T201

    # Stream the dataset (no massive download)
    ds = load_dataset(source_repo, split="train")

    # Filter
    ds = ds.filter(lambda x: len(x["text"]) > min_chars)

    # Save to disk as raw text (SAELens will handle tokenization)
    print("Materializing to disk...")  # noqa: T201

    ds.save_to_disk(dataset_path)

    print(f"Saved {len(ds)} documents to {dataset_path}")  # noqa: T201

device = "cuda"

batch_size = 4096
total_training_steps = 30_000
total_tokens = batch_size * total_training_steps

lr_warm_up_steps = 1000
lr_decay_steps = total_training_steps // 5  # 20% of training

# So many damn parameters
cfg = LanguageModelSAERunnerConfig(
    sae=ContextTrainingSAEConfig(
        d_in=768,  # For pythia and gpt2-small,
        d_sae=8192,  # Good amount of features,
        k_context=64,  # Keep number of active features the same for testing purposes
        k_token=64,
        normalize_activations="expected_average_only_in",
    ),
    model_name="pythia-160m",
    model_class_name="HookedTransformer",
    hook_name="blocks.8.hook_resid_post",
    dataset_path=dataset_path,
    is_dataset_tokenized=False,
    # Training Parameters
    lr=3e-4,
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_decay_steps,
    train_batch_size_tokens=batch_size,
    dtype="bfloat16",
    device=device,
    context_size=2048,
    disable_concat_sequences=True,
    training_tokens=total_tokens,
    n_batches_in_buffer=1024,
    store_batch_size_prompts=512,
    # Wandb
    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="Context SAE test",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
    ),
    n_checkpoints=3,
    checkpoint_path="checkpoints",
    # Try compilation, since we have an H100 :)
    compile_llm=True,
    compile_sae=True,
)

sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()
