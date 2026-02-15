import os

import sae_bench

# Base path to evals
evals_path = os.path.join(os.path.dirname(sae_bench.__file__), "evals")

print(f"Checking contents of: {evals_path}")
try:
    print(f"Files in evals: {os.listdir(evals_path)}")

    # Check if 'absorption' exists and what's inside
    abs_path = os.path.join(evals_path, "absorption")
    if os.path.exists(abs_path):
        print(f"Files in evals/absorption: {os.listdir(abs_path)}")
    else:
        print("!!! 'absorption' folder NOT FOUND in evals !!!")

except FileNotFoundError as e:
    print(f"Error: {e}")
