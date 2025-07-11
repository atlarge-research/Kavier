from simulator.constants import COMPUTE_EFFICIENCY, MEMORY_EFFICIENCY
from simulator.performance.util.specs import GPUSpec, LLMSpec


def get_decode_time_s(n_out: int, llm: LLMSpec, gpu: GPUSpec, kv_cache: bool) -> float:
    f_tok = 2.0 * llm.m_params
    b = llm.p_bytes

    compute_bound = f_tok / (gpu.fp_16_tensor_core_tflops * 1e12 * COMPUTE_EFFICIENCY)
    memory_bound = (b * llm.m_params) / (gpu.bandwidth_bps * MEMORY_EFFICIENCY)
    time_per_token = max(compute_bound, memory_bound)

    if kv_cache:
        return n_out * time_per_token
    else:
        return (n_out * (n_out + 1) / 2) * time_per_token
