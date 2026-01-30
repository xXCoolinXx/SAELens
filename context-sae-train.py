import torch

from sae_lens import (
    ContextTrainingSAEConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
)

torch.set_float32_matmul_precision("high")

# Silences the graph break warning
torch._dynamo.config.capture_scalar_outputs = True

# dataset_path = "./pile_long_context"
source_repo = "monology/pile-uncopyrighted"
# min_chars = 7000  # Rough proxy for 2048 tokens (approx 3.5 chars/token)
# n_docs = 100_000

# if Path(dataset_path).exists():
#     print(f"Dataset already exists at {dataset_path}")  # noqa: T201

#     dataset = datasets.load_from_disk(dataset_path, keep_in_memory=True)
# else:
#     print(f"{dataset_path} not found. Streaming and filtering...")  # noqa: T201

#     # Stream the dataset (no massive download)
#     ds = load_dataset(source_repo, split="train", streaming=True)

#     # Filter
#     long_docs = ds.filter(lambda x: len(x["text"]) > min_chars).take(n_docs)

#     # Save to disk as raw text (SAELens will handle tokenization)
#     print("Materializing to disk...")  # noqa: T201

#     def gen():  # type: ignore
#         yield from long_docs

#     # We reuse the features from the stream to avoid schema inference overhead
#     dataset = datasets.Dataset.from_generator(gen, features=long_docs.features)

#     dataset.save_to_disk(dataset_path)  # type: ignore

#     print(f"Saved {len(dataset)} documents to {dataset_path}")  # type: ignore # noqa: T201

device = "cuda"

batch_size = 4096
total_training_steps = 122_070
total_tokens = (
    batch_size * total_training_steps
)  # Should be 500M in line with SAE bench

lr_warm_up_steps = 1000
lr_decay_steps = total_training_steps // 5  # 20% of training

# So many damn parameters
cfg = LanguageModelSAERunnerConfig(
    sae=ContextTrainingSAEConfig(
        d_in=768,  # For pythia and gpt2-small,
        d_sae=16384,  # Good amount of features, compare to T-SAE paper
        k_context=4,  # Split used by T-SAE
        k_token=16,
        pct_context_features=0.2,
        aux_loss_coefficient=1 / 32,  # Following anthropic
        normalize_activations="expected_average_only_in",
    ),
    model_name="pythia-160m-deduped",  # Use deduped, apparently its more interpretable
    model_class_name="HookedTransformer",
    hook_name="blocks.8.hook_resid_post",
    dataset_path=source_repo,  # We already loaded the dataset because we have to use custom code # type: ignore
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
    store_batch_size_prompts=32,
    # Wandb
    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="Context SAE test",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=100000000000,  # Evals are broken for this arch, just don't do it for now
    ),
    n_checkpoints=3,
    checkpoint_path="checkpoints",
    context_processor="near_pairs",
    # Try compilation, since we have an H100 :)
    compile_llm=True,
    compile_sae=True,
)

sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()  # type: ignore
