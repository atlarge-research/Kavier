"""Analytical decode-stage latency model (roofline: max of compute- and memory-bound per-token time)."""

from kavier.sdk.io.constants import COMPUTE_EFFICIENCY, FLOPS_PER_PARAM_PER_TOKEN, MEMORY_EFFICIENCY
from kavier.sdk.library.specs.GPUSpec import GPUSpec
from kavier.sdk.library.specs.LLMSpec import LLMSpec
from kavier.sdk.units import FLOPS_PER_TFLOP


def get_decode_time_s(n_out: int, llm: LLMSpec, gpu: GPUSpec, kv_cache: bool) -> float:
    """Decode time (s) for ``n_out`` tokens: linear with KV cache, quadratic (n(n+1)/2) without it."""
    # Per-token cost is set by the parameters a token actually touches: active_params
    # (== m_params for dense models; only the routed experts for MoE). This holds for the
    # memory-bound term too — a MoE decode step reads just the active-expert weights.
    f_tok = FLOPS_PER_PARAM_PER_TOKEN * llm.active_params
    b = llm.p_bytes

    compute_bound = f_tok / (gpu.fp_16_tensor_core_tflops * FLOPS_PER_TFLOP * COMPUTE_EFFICIENCY)
    memory_bound = (b * llm.active_params) / (gpu.bandwidth_bps * MEMORY_EFFICIENCY)
    time_per_token = max(compute_bound, memory_bound)

    if kv_cache:
        return n_out * time_per_token
    else:
        return (n_out * (n_out + 1) / 2) * time_per_token
