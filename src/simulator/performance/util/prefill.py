from simulator.constants import COMPUTE_EFFICIENCY, PREFILL_OVERHEAD_S
from simulator.performance.util.specs import GPUSpec, LLMSpec


def get_prefill_time_s(n_in: int, llm: LLMSpec, gpu: GPUSpec) -> float:
    """
    Computes the prefill time in milliseconds for a given number of input tokens.
    Formula: T_prefill = system_overhead + (N_in x f_tok) / f_gpu
    :param n_in: Number of input tokens.
    :param llm: LLMSpec object containing model specifications.
    :param gpu: GPUSpec object containing GPU specifications.
    :return: Prefill time in milliseconds.
    """
    f_tok: float = 2.0 * llm.m_params  # FLOPs per token linearly scale with model size [2]
    f_gpu: float = gpu.fp_16_tensor_core_tflops * 1e12 * COMPUTE_EFFICIENCY
    return PREFILL_OVERHEAD_S + (n_in * f_tok) / f_gpu
