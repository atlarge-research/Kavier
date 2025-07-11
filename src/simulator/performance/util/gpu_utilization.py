from simulator.constants import MAX_GPU_UTILIZATION


def get_gpu_utilization(
    t: float, t_prefill, t_decode, warm: float = 0.2, cool: float = 0.2
) -> float:
    """
    First 100ms and last 100ms keep the GPU at around 50%, and memory at around 50%.
    During the prefill and decode phase, both, GPU and memory reach the system's cap (e.g., 85%).
    """
    if t < warm:
        return 0.5

    if t < t_prefill + t_decode - cool:
        return MAX_GPU_UTILIZATION

    return 0.5
