from kavier_io.constants import MAX_GPU_UTILIZATION


def get_gpu_utilization(t: float, t_prefill, t_decode, warm: float = 0.2, cool: float = 0.2) -> float:
    if t < warm:
        return 0.5

    if t < t_prefill + t_decode - cool:
        return MAX_GPU_UTILIZATION

    return 0.5
