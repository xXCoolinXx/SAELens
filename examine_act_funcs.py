import matplotlib.pyplot as plt
import numpy as np

from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.load_from_disk(
    # path="/scratch/Collin/SAELens/checkpoints/omyz0sxn/375001088",
    # path="/scratch/Collin/SAELens/checkpoints/uvvum8hk/118501376",
    # path="/scratch/Collin/SAELens/checkpoints/b6bjfcff/final_250003456",
    path="/scratch/Collin/SAELens/checkpoints/i83pn1tq/final_250003456",
    device="cuda",
)

# Get the PWL activation module
pwl = sae.activation_fn.pwl  # adjust attribute path if needed

# Get knot positions and values
knots = pwl.knots.detach().cpu().numpy()  # (n_knots,)
values = pwl.values.detach().cpu().numpy()  # (n_features, n_knots)

n_features, n_knots = values.shape

# Compute what ReLU would give at each knot
relu_values = np.maximum(knots, 0.0)  # (n_knots,)

# L2 deviation from ReLU for each feature
deviations = np.sqrt(
    np.mean((values - relu_values[None, :]) ** 2, axis=1)
)  # (n_features,)

# Get top 20 most deviant
top_k = 20
top_indices = np.argsort(deviations)[-top_k:][::-1]

# Also get 5 least deviant for comparison
bottom_k = 5
bottom_indices = np.argsort(deviations)[:bottom_k]

# Plot top deviants
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for i, feat_idx in enumerate(top_indices):
    ax = axes[i]
    ax.plot(knots, values[feat_idx], "b-o", markersize=3, label="Learned")
    ax.plot(knots, relu_values, "r--", alpha=0.5, label="ReLU")
    ax.set_title(f"Feature {feat_idx}\nL2 dev: {deviations[feat_idx]:.4f}")
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle("Top 20 Most Deviant Activation Functions", fontsize=16)
plt.tight_layout()
plt.savefig("top_deviant_activations.png", dpi=150, bbox_inches="tight")
plt.show()

# Plot least deviant for comparison
fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4))
for i, feat_idx in enumerate(bottom_indices):
    ax = axes2[i]
    ax.plot(knots, values[feat_idx], "b-o", markersize=3, label="Learned")
    ax.plot(knots, relu_values, "r--", alpha=0.5, label="ReLU")
    ax.set_title(f"Feature {feat_idx}\nL2 dev: {deviations[feat_idx]:.4f}")
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)

plt.suptitle("5 Least Deviant (Most ReLU-like) Activation Functions", fontsize=14)
plt.tight_layout()
plt.savefig("least_deviant_activations.png", dpi=150, bbox_inches="tight")
plt.show()

# Print summary statistics
print(f"Deviation stats across {n_features} features:")
print(f"  Mean: {deviations.mean():.4f}")
print(f"  Median: {np.median(deviations):.4f}")
print(f"  Std: {deviations.std():.4f}")
print(f"  Min: {deviations.min():.4f}")
print(f"  Max: {deviations.max():.4f}")
print(f"  90th percentile: {np.percentile(deviations, 90):.4f}")
print(f"  95th percentile: {np.percentile(deviations, 95):.4f}")
print(f"  99th percentile: {np.percentile(deviations, 99):.4f}")

# Histogram of deviations
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.hist(deviations, bins=50, edgecolor="black", alpha=0.7)
ax3.set_xlabel("L2 Deviation from ReLU")
ax3.set_ylabel("Count")
ax3.set_title("Distribution of Activation Function Deviations from ReLU")
ax3.axvline(
    x=np.median(deviations),
    color="r",
    linestyle="--",
    label=f"Median: {np.median(deviations):.4f}",
)
ax3.legend()
plt.tight_layout()
plt.savefig("deviation_histogram.png", dpi=150, bbox_inches="tight")
plt.show()
