"""Kavier public API — usage examples.

`import kavier` exposes two workload namespaces — `inference` and `training` —
each with the same four predictors:

    kavier.inference.performance / energy / efficiency / carbon
    kavier.training .performance / energy / efficiency / carbon

Every predictor takes a *batch* (a pandas DataFrame, a list of dicts, or a single
dict — one row per workload) and returns a DataFrame: your input rows plus the
predicted columns. The numbers match the `kavier` CLIs and UI, because all three
call this same API.

Run with:  python docs/usage.py
"""

import pandas as pd

import kavier

# --------------------------------------------------------------------------- #
# 1) Inference — a batch of serving workloads
# --------------------------------------------------------------------------- #
# Each row: which model on which GPU, and the request shape.
# NB: in the API, `input_tokens` / `output_tokens` are scalar *counts* — unlike trace CSVs,
# where `input_tokens` is the token-id *list* and `num_input_tokens` is the count.
inference_batch = pd.DataFrame(
    [
        {"model": "Llama-3-8B", "gpu": "A10", "num_requests": 128, "input_tokens": 512, "output_tokens": 128},
        {
            "model": "mistral-7b-v0.1",
            "gpu": "NVIDIA-A100-SXM4-80GB",
            "num_requests": 256,
            "input_tokens": 1024,
            "output_tokens": 256,
        },
    ]
)

performance = kavier.inference.performance(inference_batch)  # + p50_ms, throughput_tok_s, gpu util
energy = kavier.inference.energy(inference_batch)  # + energy_wh, energy_per_mtoken_wh
efficiency = kavier.inference.efficiency(inference_batch)  # + financial_per_mtoken ($/Mtoken)
carbon = kavier.inference.carbon(inference_batch)  # + carbon_per_mtoken_g (gCO2)

print(performance)

# --------------------------------------------------------------------------- #
# 2) Training — a batch of fine-tuning jobs
# --------------------------------------------------------------------------- #
# Size each job by total_tokens, OR by epochs x dataset_tokens.
training_batch = pd.DataFrame(
    [
        {
            "model": "mistral-7b-v0.1",
            "gpu": "NVIDIA-A100-SXM4-80GB",
            "method": "lora",
            "batch_size": 4,
            "seq_len": 1024,
            "num_gpus": 8,
            "num_nodes": 1,
            "epochs": 3,
            "dataset_tokens": 5_000_000,
        },
        {
            "model": "Llama-2-13B",
            "gpu": "NVIDIA-A100-SXM4-80GB",
            "method": "full",
            "batch_size": 2,
            "seq_len": 2048,
            "num_gpus": 16,
            "num_nodes": 2,
            "total_tokens": 50_000_000,
        },
    ]
)

train_performance = kavier.training.performance(training_batch)  # + train_tokens_per_second, train_runtime
train_energy = kavier.training.energy(training_batch)
train_efficiency = kavier.training.efficiency(training_batch)
train_carbon = kavier.training.carbon(training_batch)

print(train_performance)

# --------------------------------------------------------------------------- #
# 3) Inputs are flexible — a single dict or a list of dicts works too
# --------------------------------------------------------------------------- #
one = kavier.inference.performance(
    {"model": "Llama-3-8B", "gpu": "A10", "num_requests": 64, "input_tokens": 256, "output_tokens": 64}
)
print(one)
