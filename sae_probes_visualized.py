import json

import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA
# ==========================================


def load_results(path):
    with open(path) as f:
        data = json.load(f)["eval_result_details"]
    return {d["dataset_name"]: d for d in data}


# name = "pythia-160m-deduped_layer_8"
name = "gemma-2-2b_layer_12"
extra = "20_20"

context_data = load_results(
    f"./eval_results/sparse_probing_sae_probes/{name}_identity_sae_custom_sae_0_8191_eval_results.json"
)
token_data = load_results(
    f"./eval_results/sparse_probing_sae_probes/{name}_identity_sae_custom_sae_8192_16383_eval_results.json"
)
combined_data = load_results(
    f"./eval_results/sparse_probing_sae_probes/{name}_identity_sae_custom_sae_0_16383_eval_results.json"
)

with open("probe_categories.json") as f:
    categories = json.load(f)

k_values = [1, 2, 5, 10, 20, 50, 100]
sources = {"Context": context_data, "Token": token_data, "Combined": combined_data}
colors = {"Context": "blue", "Token": "orange", "Combined": "green"}

# ==========================================
# 2. CALCULATE AVERAGES
# ==========================================


# Helper function to get averages for SAE results
def get_averages(dataset_list, source_dict):
    sums = {k: {"acc": 0.0, "f1": 0.0, "auc": 0.0} for k in k_values}
    count = 0

    for ds_name in dataset_list:
        if ds_name not in source_dict:
            continue
        count += 1
        row = source_dict[ds_name]
        for k in k_values:
            sums[k]["acc"] += row.get(f"sae_top_{k}_test_accuracy", 0)
            sums[k]["f1"] += row.get(f"sae_top_{k}_test_f1", 0)
            sums[k]["auc"] += row.get(f"sae_top_{k}_test_auc", 0)

    # Calculate means
    averages = {"acc": [], "f1": [], "auc": []}
    if count > 0:
        for k in k_values:
            averages["acc"].append(sums[k]["acc"] / count)
            averages["f1"].append(sums[k]["f1"] / count)
            averages["auc"].append(sums[k]["auc"] / count)
    else:
        zeros = [0] * len(k_values)
        averages = {"acc": zeros, "f1": zeros, "auc": zeros}

    return averages


# Helper function to get averages for the Original Linear Probe (Baseline)
# We only need to pull this from one source (e.g., context_data) as values are identical
def get_baseline_averages(dataset_list, source_dict):
    sums = {"acc": 0.0, "f1": 0.0, "auc": 0.0}
    count = 0

    for ds_name in dataset_list:
        if ds_name not in source_dict:
            continue
        count += 1
        row = source_dict[ds_name]
        # Pulling the standard linear probe metrics
        sums["acc"] += row.get("llm_test_accuracy", 0)
        sums["f1"] += row.get("llm_test_f1", 0)
        sums["auc"] += row.get("llm_test_auc", 0)

    if count > 0:
        return {k: v / count for k, v in sums.items()}
    return {"acc": 0.0, "f1": 0.0, "auc": 0.0}


# Prepare data structure for plotting
all_plots = []

for high_level_cat, subcategories in categories.items():
    for sub_cat, dataset_list in subcategories.items():
        entry = {"title": f"{sub_cat}\n({high_level_cat})", "lines": {}}

        # Calculate stats for Context, Token, Combined for this subcategory
        for source_name, source_dict in sources.items():
            entry["lines"][source_name] = get_averages(dataset_list, source_dict)

        # Calculate baseline stats (using context_data as the reference)
        entry["baseline"] = get_baseline_averages(dataset_list, context_data)

        all_plots.append(entry)

# ==========================================
# 3. PLOTTING
# ==========================================

num_rows = len(all_plots)
fig, axes = plt.subplots(num_rows, 3, figsize=(18, 4 * num_rows), squeeze=False)

metrics_map = [("acc", "Accuracy"), ("f1", "F1 Score"), ("auc", "AUROC")]

for row_idx, plot_info in enumerate(all_plots):
    for col_idx, (metric_key, metric_label) in enumerate(metrics_map):
        ax = axes[row_idx, col_idx]

        # 1. Plot the Baseline (Horizontal Red Line)
        baseline_value = plot_info["baseline"][metric_key]
        ax.axhline(
            y=baseline_value,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Original Probe",
        )

        # 2. Plot the 3 lines (Context, Token, Combined)
        for source_name, data in plot_info["lines"].items():
            y_values = data[metric_key]
            ax.plot(
                k_values,
                y_values,
                marker="o",
                markersize=4,
                label=source_name,
                color=colors[source_name],
            )

        # Formatting
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.set_xticklabels(k_values)
        ax.grid(True, which="both", ls="-", alpha=0.2)

        # Set titles and labels
        if row_idx == 0:
            ax.set_title(metric_label, fontsize=14, fontweight="bold")

        if col_idx == 0:
            ax.set_ylabel(plot_info["title"], fontsize=10, fontweight="bold")

        # Only add legend to the top-left plot to reduce clutter
        if row_idx == 0 and col_idx == 0:
            ax.legend()

plt.tight_layout()
plt.savefig(f"{name}_{extra}_comparison_subplots.png", dpi=150)
plt.show()
