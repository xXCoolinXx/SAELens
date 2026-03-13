import torch

from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
    SMIXAETrainingConfig,
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
total_tokens = 500_000_000
# total_tokens = 1000 * batch_size

total_training_steps = total_tokens // batch_size

lr_warm_up_steps = 1000
lr_decay_steps = total_training_steps // 5  # 20% of training

# So many damn parameters
cfg = LanguageModelSAERunnerConfig(
    sae=SMIXAETrainingConfig(
        d_in=2304,  # d_in=768,  # For pythia and gpt2-small,
        n_experts=2048,  # Good amount of features, compare to Gemma Scope
        d_expert=16,
        d_bottleneck=3,
        d_sae=16 * 2048,  # this parameter is ignored,
        grump_init=0.01,
        grump_bandwidth=2.0,
        pre_act_loss_coefficient=3e-6,
        grump_tanh_coefficient=4,
        l0_coefficient=10,
        rescale_acts_by_decoder_norm=True,
        normalize_activations="expected_average_only_in",
    ),
    model_name="gemma-2-2b",  # "pythia-160m-deduped",  # Use deduped, apparently its more interpretable
    model_class_name="HookedTransformer",
    hook_name="blocks.12.hook_resid_post",  # "blocks.8.hook_resid_post",
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
    n_batches_in_buffer=40,
    store_batch_size_prompts=32,
    # Wandb
    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="SMIXAE on Gemma 2-2B, Batch Top K",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=5000000,
    ),
    n_checkpoints=3,
    checkpoint_path="checkpoints",
    context_processor=None,
    # Try compilation, since we have an H100 :)
    compile_llm=True,
    compile_sae=True,
    save_final_checkpoint=True,
)

sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()  # type: ignore
