"""Analytical prefill-stage latency model (compute-bound FLOPs over effective GPU throughput plus fixed overhead)."""

from kavier.sdk.io.constants import COMPUTE_EFFICIENCY, FLOPS_PER_PARAM_PER_TOKEN, PREFILL_OVERHEAD_S
from kavier.sdk.library.specs.GPUSpec import GPUSpec
from kavier.sdk.library.specs.LLMSpec import LLMSpec
from kavier.sdk.units import FLOPS_PER_TFLOP


def get_prefill_time_s(n_in: int, llm: LLMSpec, gpu: GPUSpec) -> float:
    """Prefill time (s) for ``n_in`` tokens: PREFILL_OVERHEAD_S + compute-bound (2*active_params FLOPs/tok) / TFLOPS."""
    # FLOPs per token scale with the parameters touched per token [2]: active_params
    # (== m_params for dense models; only the routed experts for MoE).
    f_tok: float = FLOPS_PER_PARAM_PER_TOKEN * llm.active_params
    f_gpu: float = gpu.fp_16_tensor_core_tflops * FLOPS_PER_TFLOP * COMPUTE_EFFICIENCY
    return PREFILL_OVERHEAD_S + (n_in * f_tok) / f_gpu
