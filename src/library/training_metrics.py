"""
Training metrics definitions for validation.
"""

# Performance metrics tracked during training
TRAINING_PERFORMANCE_METRICS = [
    "dataset_tokens_per_second",
    "train_runtime",
    "gpu_compute_utilization_avg",
]

# Energy metrics tracked during training
TRAINING_ENERGY_METRICS = [
    "gpu_power_watts_avg",
    "total_energy_wh",
]

# Made with Bob
