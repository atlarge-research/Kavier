"""Analytical prefill-stage latency model (compute-bound FLOPs over effective GPU throughput plus fixed overhead)."""

from kavier_io.constants import COMPUTE_EFFICIENCY, PREFILL_OVERHEAD_S
from kavier_library.specs.GPUSpec import GPUSpec
from kavier_library.specs.LLMSpec import LLMSpec


def get_prefill_time_s(n_in: int, llm: LLMSpec, gpu: GPUSpec) -> float:
    """Prefill time (s) for ``n_in`` tokens: ``PREFILL_OVERHEAD_S`` + compute-bound (2*params FLOPs/tok) / TFLOPS."""
    f_tok: float = 2.0 * llm.m_params  # FLOPs per token scale linearly with model size [2]
    f_gpu: float = gpu.fp_16_tensor_core_tflops * 1e12 * COMPUTE_EFFICIENCY
    return PREFILL_OVERHEAD_S + (n_in * f_tok) / f_gpu
