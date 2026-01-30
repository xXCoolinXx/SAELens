import json
import os

import numpy as np


def analyze_partition_scores(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading results from: {file_path}")
    with open(file_path) as f:
        data = json.load(f)

    # extract architecture details to be safe
    sae_cfg = data.get("sae_cfg_dict", {})
    d_sae = sae_cfg.get("d_sae", 16384)

    # Calculate split point
    split_point = int(d_sae * 0.2)

    print(f"Total SAE Latents: {d_sae}")
    print(
        f"Split Point: {split_point} (First half < {split_point}, Second half >= {split_point})"
    )

    # Get the dictionary of individual feature results
    # Structure is "latent_index_string": { "score": float, ... }
    results = data.get("eval_result_unstructured", {})
    print(f"Number of features scored in this run: {len(results)}\n")

    first_half_scores = []  # Likely Shared/Context
    second_half_scores = []  # Likely Token

    for latent_id_str, details in results.items():
        try:
            latent_id = int(latent_id_str)
            score = details.get("score")

            # Skip if score is missing/null
            if score is None:
                continue

            if latent_id < split_point:
                first_half_scores.append(score)
            else:
                second_half_scores.append(score)

        except ValueError:
            print(f"Skipping invalid key: {latent_id_str}")
            continue

    # Calculate Statistics
    avg_first = np.mean(first_half_scores) if first_half_scores else 0.0
    avg_second = np.mean(second_half_scores) if second_half_scores else 0.0

    std_first = np.std(first_half_scores) if first_half_scores else 0.0
    std_second = np.std(second_half_scores) if second_half_scores else 0.0

    print("=" * 60)
    print(f"PARTITION 1 (Indices 0 - {split_point - 1}) [Shared/Context]")
    print("=" * 60)
    print(f"Count: {len(first_half_scores)}")
    print(f"Mean Score: {avg_first:.4f}")
    print(f"Std Dev:    {std_first:.4f}")

    print("\n" + "=" * 60)
    print(f"PARTITION 2 (Indices {split_point} - {d_sae - 1}) [Token]")
    print("=" * 60)
    print(f"Count: {len(second_half_scores)}")
    print(f"Mean Score: {avg_second:.4f}")
    print(f"Std Dev:    {std_second:.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    diff = avg_second - avg_first
    print(f"Difference (Partition 2 - Partition 1): {diff:+.4f}")
    if diff > 0:
        print(">> The second half (Token) scored higher.")
    else:
        print(">> The first half (Shared) scored higher.")


def analyze_explanation_content(file_path):
    with open(file_path) as f:
        data = json.load(f)

    sae_cfg = data.get("sae_cfg_dict", {})
    d_sae = sae_cfg.get("d_sae", 16384)
    split_point = int(d_sae * 0.2)

    results = data.get("eval_result_unstructured", {})

    # Keywords to look for
    context_keywords = {
        # "syntax",
        "structure",
        "pattern",
        "context",
        "topic",
        "section",
        "header",
        "citation",
        "reference",
        "legal",
        "medical",
        "technical",
        "algebraic",
        "equation",
        "format",
        # "indent",
        "bracket",
        "parentheses",
        "symbol",
        "notation",
        "expression",
        "phrase",
        "clause",
        "sentence",
        "grammar",
        "preposition",
        # "conjunction",
        # "verb",
        # "noun",
        # "adjective",
        # "tense",
        # "plural",
        # "singular",
        # "capitalized",
        # "uppercase",
        # "lowercase",
        # "sequence",
        "series",
        "list",
        "range",
        "span",
        "duration",
        "following",
        "preceding",
        "start",
        "end",
        "middle",
        "between",
        "surrounding",
        "related to",
        "pertaining to",
        "involving",
        "describing",
        "indicating",
        "expressing",
    }

    # Keywords implying SPECIFIC TOKEN MATCHING
    # "fires on the word 'the'", "fires on the substring 'ing'", "fires on the number '1'"
    token_keywords = {
        "word",
        "string",
        "substring",
        "token",
        "character",
        "letter",
        "digit",
        "number",
        "literal",
        "specific",
        "exact",
        "match",
        "consisting of",
        "containing",
        "starts with",
        "ends with",
        "followed by",
        "preceded by",
        # Note: "followed by" is tricky, it implies sequence but often specifies exact tokens.
        # I'll keep it here as it usually denotes n-grams.
    }

    # Counters
    shared_context_hits = 0
    token_context_hits = 0
    shared_token_hits = 0
    token_token_hits = 0

    shared_explanations = []
    token_explanations = []

    for latent_id_str, details in results.items():
        latent_id = int(latent_id_str)
        explanation = details.get("explanation", "").lower()

        if not explanation:
            continue

        # Classify based on partition
        is_shared = latent_id < split_point

        if is_shared:
            shared_explanations.append(explanation)
            if any(k in explanation for k in context_keywords):
                shared_context_hits += 1
            if any(k in explanation for k in token_keywords):
                shared_token_hits += 1
        else:
            token_explanations.append(explanation)
            if any(k in explanation for k in context_keywords):
                token_context_hits += 1
            if any(k in explanation for k in token_keywords):
                token_token_hits += 1

    n_shared = len(shared_explanations)
    n_token = len(token_explanations)

    print(f" Analyzed {n_shared} Shared features and {n_token} Token features.")
    print("=" * 60)
    print(" KEYWORD ANALYSIS")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Shared':<10} | {'Token':<10}")
    print("-" * 56)

    print(
        f"{'Has Context Words':<30} | {shared_context_hits / n_shared:.1%}      | {token_context_hits / n_token:.1%}"
    )
    print("('follow', 'precede', 'topic', 'syntax', etc)")
    print("-" * 56)

    print(
        f"{'Has Token Words':<30} | {shared_token_hits / n_shared:.1%}      | {token_token_hits / n_token:.1%}"
    )
    print("('word', 'string', 'capital', etc)")

    print("\n" + "=" * 60)
    print(" RANDOM EXAMPLES (Shared)")
    print("=" * 60)
    import random

    for ex in random.sample(shared_explanations, 5):
        print(f"- {ex}")

    print("\n" + "=" * 60)
    print(" RANDOM EXAMPLES (Token)")
    print("=" * 60)
    for ex in random.sample(token_explanations, 5):
        print(f"- {ex}")


# Run the analysis
file_path = "./eval_results/autointerp/pythia-160m-deduped_layer_8_identity_sae_custom_sae_eval_results.json"
analyze_partition_scores(file_path)
analyze_explanation_content(file_path)
