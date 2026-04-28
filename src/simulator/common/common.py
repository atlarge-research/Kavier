"""
Common constants and utilities for Kavier simulation.

Clearly distinguishes between inference and training (fine-tuning) metrics.
"""

# ============================================================================
# INFERENCE METRICS (from kavier-perf)
# ============================================================================

INFERENCE_PERFORMANCE_METRICS = [
    "prefill_latency_ms",
    "decode_latency_ms",
    "throughput_tokens_per_second",
    "gpu_utilization_avg",
]

INFERENCE_ENERGY_METRICS = [
    "gpu_power_watts_avg",
    "total_energy_wh",
]

# ============================================================================
# TRAINING (FINE-TUNING) METRICS (from kavier-train)
# ============================================================================

TRAINING_PERFORMANCE_METRICS = [
    "dataset_tokens_per_second",      # Throughput during fine-tuning
    "train_runtime",                  # Total training time (seconds)
    "gpu_compute_utilization_avg",   # GPU compute utilization (%)
]

TRAINING_ENERGY_METRICS = [
    "gpu_power_watts_avg",           # Instant power consumption (W)
    "total_energy_wh",               # Total energy consumption (Wh)
]

# ============================================================================
# VALIDATION METRICS
# ============================================================================

ALL_TRAINING_VALIDATION_METRICS = TRAINING_PERFORMANCE_METRICS + TRAINING_ENERGY_METRICS
ALL_INFERENCE_VALIDATION_METRICS = INFERENCE_PERFORMANCE_METRICS + INFERENCE_ENERGY_METRICS

# Made with Bob
