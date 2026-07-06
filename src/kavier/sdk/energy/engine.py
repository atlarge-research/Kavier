"""kavier.sdk.energy engine: GPU power models (analytical MSE curve + empirical measured-table lookup)."""

from __future__ import annotations

from kavier.sdk.library.specs.GPUSpec import GPUSpec


def mse_power(compute_utilization: float, memory_utilization: float, gpu: GPUSpec) -> float:
    """Analytical power: idle + (max-idle)*(2u - u^r); u = max(compute, mem) util, r = gpu.mse_calib_factor."""
    u = max(min(max(compute_utilization, memory_utilization), 1.0), 0.0)
    if u <= 0.0:
        return float(gpu.idle_power_w)
    r = gpu.mse_calib_factor
    return float(gpu.idle_power_w + (gpu.max_power_w - gpu.idle_power_w) * (2.0 * u - u**r))


def empirical_power(compute_utilization: float, memory_utilization: float, gpu: GPUSpec) -> float:
    """Power from a measured utilization->watts table per GPU; needs measured data (none wired yet)."""
    raise NotImplementedError("empirical power model needs measured utilization->power data per GPU; none provided yet")
