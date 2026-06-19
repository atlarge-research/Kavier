"""Piecewise GPU-utilization model over a request's lifetime (warm-up / steady / cool-down)."""

from kavier_io.constants import MAX_GPU_UTILIZATION


def get_gpu_utilization(t: float, t_prefill, t_decode, warm: float = 0.2, cool: float = 0.2) -> float:
    """Return fractional GPU utilization at elapsed time ``t`` (s): 0.5 during the warm/cool windows, ``MAX_GPU_UTILIZATION`` in between."""
    if t < warm:
        return 0.5

    if t < t_prefill + t_decode - cool:
        return MAX_GPU_UTILIZATION

    return 0.5
