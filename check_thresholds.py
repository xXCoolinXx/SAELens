from safetensors.torch import load_file

sd = load_file(
    "checkpoints/omyz0sxn/375001088/sae_weights.safetensors"
)  # adjust filename
print([f"{k}, {v}" for k, v in sd.items() if "threshold" in k])
