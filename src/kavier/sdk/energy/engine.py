"""kavier.sdk.energy engine: the analytical MSE-curve GPU power model."""

from __future__ import annotations

from kavier.sdk.library.specs.GPUSpec import GPUSpec


def mse_power(compute_utilization: float, memory_utilization: float, gpu: GPUSpec) -> float:
    """Analytical power: idle + (max-idle)*(2u - u^r); u = max(compute, mem) util, r = gpu.mse_calib_factor."""
    u = max(min(max(compute_utilization, memory_utilization), 1.0), 0.0)
    if u <= 0.0:
        return float(gpu.idle_power_w)
    r = gpu.mse_calib_factor
    return float(gpu.idle_power_w + (gpu.max_power_w - gpu.idle_power_w) * (2.0 * u - u**r))
