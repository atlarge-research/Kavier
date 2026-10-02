"""Show the Kavier Python API on inference and training batches.

`import kavier` has two workload namespaces, `inference` and `training`, each with
four predictors:

    kavier.inference.performance / energy / efficiency / carbon
    kavier.training.performance / energy / efficiency / carbon

A predictor takes a batch, one row per workload, as a pandas DataFrame, a list of
dicts, or a single dict. It returns the input rows plus the predicted columns.
The `kavier` CLI calls the same API and gives the same numbers.

Run with: python docs/usage.py
"""

import pandas as pd

import kavier

# 1) Inference: a batch of serving workloads.
# `input_tokens` and `output_tokens` are counts here. In trace CSVs, `input_tokens` is the
# token-id list and `num_input_tokens` the count.
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

performance = kavier.inference.performance(inference_batch)  # + p50_ms, p95_ms, throughput_tok_s
energy = kavier.inference.energy(inference_batch)  # + energy_wh, energy_per_mtoken_wh
efficiency = kavier.inference.efficiency(inference_batch)  # + financial_per_mtoken ($/Mtoken)
carbon = kavier.inference.carbon(inference_batch)  # + carbon_per_mtoken_g (gCO2)

print(performance)

# 2) Training: a batch of fine-tuning jobs.
# Size each job by total_tokens, or by epochs x dataset_tokens.
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
            "model": "granite-3.3-8b",
            "gpu": "NVIDIA-A100-SXM4-80GB",
            "method": "full",
            "batch_size": 2,
            "seq_len": 2048,
            "num_gpus": 8,  # GPUs per node
            "num_nodes": 1,
            "total_tokens": 50_000_000,
        },
    ]
)

train_performance = kavier.training.performance(training_batch)  # + train_tokens_per_second, train_runtime
train_energy = kavier.training.energy(training_batch)
train_efficiency = kavier.training.efficiency(training_batch)
train_carbon = kavier.training.carbon(training_batch)

print(train_performance)

# 3) A single dict or a list of dicts also works.
one = kavier.inference.performance(
    {"model": "Llama-3-8B", "gpu": "A10", "num_requests": 64, "input_tokens": 256, "output_tokens": 64}
)
print(one)
