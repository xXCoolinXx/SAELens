import os
from typing import Any

import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.autointerp.main as autointerp
import sae_bench.evals.core.main as core
import sae_bench.evals.ravel.main as ravel
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.evals.sparse_probing_sae_probes.main as sparse_probing_sae_probes
import sae_bench.evals.unlearning.main as unlearning
import sae_bench.sae_bench_utils.general_utils as general_utils
import sklearn.utils.validation
import torch
from dotenv import load_dotenv
from openai import OpenAI
from openai.resources.chat import Completions
from tqdm import tqdm

from sae_lens import SAE

# Monkey patch for bfloat16 in sklearn
# 1. Capture the original validation function
# This function is the "front door" for almost all sklearn algorithms
original_check_array = sklearn.utils.validation.check_array


# 2. Define the patch
def patched_check_array(array, *args, **kwargs):
    """
    Universal interceptor.
    If data arriving at sklearn is a BFloat16 Tensor,
    convert it to Float32 before sklearn touches it.
    """
    if isinstance(array, torch.Tensor) and array.dtype == torch.bfloat16:
        # Optional: Print purely for sanity checking if you want to see it working
        # print("Sanity Patch: Converting BFloat16 -> Float32 for Sklearn")
        array = array.to(dtype=torch.float32)

    return original_check_array(array, *args, **kwargs)


# 3. Apply the patch
sklearn.utils.validation.check_array = patched_check_array

# Patch the library - for some reason they have model_name = LLM_MODEL_NAME[model_name] - stupid af
ravel.LLM_NAME_MAP["pythia-160m"] = "EleutherAI/pythia-160m"

# 1. SETTINGS
load_dotenv()
BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
TARGET_MODEL = "Qwen/Qwen3-32B"

# 2. PATCH CLIENT INITIALIZATION
# This ensures any new client = OpenAI() call uses Groq credentials
# regardless of what arguments are passed to it.
original_init = OpenAI.__init__


def patched_init(self, *args, **kwargs):  # type: ignore
    # Force the base_url and api_key
    kwargs["base_url"] = BASE_URL
    kwargs["api_key"] = API_KEY
    original_init(self, *args, **kwargs)


OpenAI.__init__ = patched_init

# 3. PATCH CHAT COMPLETION CREATION
# This intercepts the API call to swap the model parameter.
original_create = Completions.create


def patched_create(self, *args, **kwargs):  # type: ignore
    # Log what we are intercepting (optional)
    # requested_model = kwargs.get("model", "unknown")
    # print(f"Intercepted request for model: {requested_model}")  # noqa: T201
    # print(f"Redirecting to: {TARGET_MODEL}")  # noqa: T201

    # Force the model to Groq Llama 3.3
    kwargs["model"] = TARGET_MODEL

    kwargs["extra_body"] = {
        # "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # Execute the original method with modified arguments
    return original_create(self, *args, **kwargs)


Completions.create = patched_create  # type: ignore

RANDOM_SEED = 42

MODEL_CONFIGS = {
    "pythia-160m-deduped": {
        "batch_size": 512,
        "dtype": "bfloat16",
        "layers": [8],
        "d_model": 768,
    },
    "gemma-2-2b": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "layers": [12],
        "d_model": 2304,
    },
    "pythia-160m": {
        "batch_size": 256,  # Adjusted for 160M (can go higher if you have high VRAM)
        "dtype": "bfloat16",
        "layers": [8],  # The layer you trained on
        "d_model": 768,
    },
}

output_folders = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    "sparse_probing_sae_probes": "eval_results/sparse_probing_sae_probes",
    "unlearning": "eval_results/unlearning",
    "ravel": "eval_results/ravel",
}


def run_evals(
    model_name: str,
    selected_saes: list[tuple[str, Any]],
    llm_batch_size: int,
    llm_dtype: str,
    device: str,
    eval_types: list[str],
    api_key: str | None = None,
    force_rerun: bool = False,
    save_activations: bool = False,
):
    """Run selected evaluations for the given model and SAEs."""

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    # Mapping of eval types to their functions and output paths
    eval_runners = {
        "absorption": (
            lambda: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/absorption",
                force_rerun,
            )
        ),
        "autointerp": (
            lambda: autointerp.run_eval(
                autointerp.AutoInterpEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                api_key,  # type: ignore
                "eval_results/autointerp",
                force_rerun,
            )
        ),
        # TODO: Do a better job of setting num_batches and batch size
        "core": (
            lambda: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder="eval_results/core",
                verbose=True,
                dtype=llm_dtype,
                device=device,
            )
        ),
        "ravel": (
            lambda: ravel.run_eval(
                ravel.RAVELEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size // 4,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/ravel",
                force_rerun,
            )
        ),
        "scr": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    perform_scr=True,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results",  # We add scr or tpp depending on perform_scr
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "tpp": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    perform_scr=False,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results",  # We add scr or tpp depending on perform_scr
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "sparse_probing": (
            lambda: sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/sparse_probing",
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "sparse_probing_sae_probes": (
            lambda: sparse_probing_sae_probes.run_eval(
                sparse_probing_sae_probes.SparseProbingSaeProbesEvalConfig(
                    model_name=model_name,
                ),
                selected_saes,
                device,
                "eval_results/sparse_probing_sae_probes",
                force_rerun,
            )
        ),
        "unlearning": (
            lambda: unlearning.run_eval(
                unlearning.UnlearningEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_dtype=llm_dtype,
                    llm_batch_size=llm_batch_size // 8,
                ),
                selected_saes,
                device,
                "eval_results/unlearning",
                force_rerun,
            )
        ),
    }

    # Run selected evaluations
    for eval_type in tqdm(eval_types, desc="Evaluations"):
        if eval_type == "autointerp" and api_key is None:
            print("Skipping autointerp evaluation due to missing API key")  # noqa: T201
            continue
        if eval_type == "unlearning":
            if model_name != "gemma-2-2b":
                print("Skipping unlearning evaluation for non-GEMMA model")  # noqa: T201
                continue
            print("Skipping, need to clean up unlearning interface")  # noqa: T201
            continue  # TODO:
            if not os.path.exists(
                "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
            ):
                print(  # noqa: T201
                    "Skipping unlearning evaluation due to missing bio-forget-corpus.jsonl"
                )
                continue

        print(f"\n\n\nRunning {eval_type} evaluation\n\n\n")  # noqa: T201

        if eval_type in eval_runners:
            os.makedirs(output_folders[eval_type], exist_ok=True)
            eval_runners[eval_type]()


if __name__ == "__main__":
    device = general_utils.setup_environment()

    model_name = "pythia-160m-deduped"
    d_model = MODEL_CONFIGS[model_name]["d_model"]
    llm_batch_size = MODEL_CONFIGS[model_name]["batch_size"]
    llm_dtype = MODEL_CONFIGS[model_name]["dtype"]

    # Note: Unlearning is not recommended for models with < 2B parameters and we recommend an instruct tuned model
    # Unlearning will also require requesting permission for the WMDP dataset (see unlearning/README.md)
    # Absorption not recommended for models < 2B parameters

    # Select your eval types here.
    eval_types = [
        # "absorption",
        # "autointerp",
        "core",
        "ravel",
        "scr",
        "tpp",
        # "sparse_probing",
        # "sparse_probing_sae_probes",
        # "unlearning",
    ]

    api_key = API_KEY

    if "unlearning" in eval_types and not os.path.exists(
        "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
    ):
        raise Exception(
            "Please download bio-forget-corpus.jsonl for unlearning evaluation"
        )

    # If evaluating multiple SAEs on the same layer, set save_activations to True
    # This will require at least 100GB of disk space
    save_activations = False

    for hook_layer in MODEL_CONFIGS[model_name]["layers"]:  # type: ignore
        sae, cfg_dict, sparsity = SAE.load_from_disk(
            path="/scratch/Collin/SAELens/checkpoints/omyz0sxn/375001088",
            device=device,
        )

        selected_saes = [(f"{model_name}_layer_{hook_layer}_identity_sae", sae)]

        for sae_name, sae in selected_saes:
            sae = sae.to(dtype=general_utils.str_to_dtype(llm_dtype))  # type: ignore
            sae.cfg.dtype = llm_dtype

        run_evals(
            model_name,
            selected_saes,
            llm_batch_size,  # type: ignore
            llm_dtype,  # type: ignore
            device,
            eval_types=eval_types,
            api_key=api_key,
            force_rerun=False,
            save_activations=False,
        )
