"""Simulation tuning and physics constants: efficiencies, overheads, utilisation cap, FLOPs per token."""

COMPUTE_EFFICIENCY = 0.30
MEMORY_EFFICIENCY = 0.60
PREFILL_OVERHEAD_S = 0.025
MAX_GPU_UTILIZATION = 0.95

# Forward-pass FLOPs per parameter per token (the 2 in 2*N*tokens). Used by the inference
# prefill/decode roofline and the training-step FLOP count.
FLOPS_PER_PARAM_PER_TOKEN = 2.0
