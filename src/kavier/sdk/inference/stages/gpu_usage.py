"""Piecewise GPU-utilization model over a request's lifetime (warm-up / steady / cool-down)."""

from kavier.sdk.io.constants import MAX_GPU_UTILIZATION


def get_gpu_utilization(t: float, t_prefill, t_decode, warm: float = 0.2, cool: float = 0.2) -> float:
    """Fractional GPU use at elapsed ``t`` (s): 0.5 in the warm/cool windows, ``MAX_GPU_UTILIZATION`` between."""
    if t < warm:
        return 0.5

    if t < t_prefill + t_decode - cool:
        return MAX_GPU_UTILIZATION

    return 0.5
