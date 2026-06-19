"""Analytical training-step engine: FLOPs/MFU + comm/optimizer model yielding throughput, runtime, GPU utilization and power."""

from __future__ import annotations

import math
from typing import Any, Dict

from kavier_library.lookup import get_gpu, get_llm
from kavier_library.specs.GPUSpec import GPUSpec
from kavier_training.core.calibration import (
    get_comm_scale,
    get_interaction_scale,
    get_method_scale,
    get_mfu_batch_scale,
    get_mfu_multiplier,
    get_model_scale,
    get_multi_gpu_correction,
    get_training_overhead_s,
)
from kavier_training.core.config import INFINIBAND_GBPS


def _compute_mfu(batch_size: int, gpu: GPUSpec, calibrated: bool = True) -> float:
    base = gpu.mfu_factor * (get_mfu_multiplier(gpu.name) if calibrated else 1.0)
    alpha, beta = get_mfu_batch_scale()
    batch_scale = min(1.0, alpha * math.log2(batch_size) + beta)
    return float(base * batch_scale)


def _calculate_gpu_power(
    compute_utilization: float,
    memory_utilization: float,
    gpu_spec: GPUSpec,
) -> float:
    u = max(min(max(compute_utilization, memory_utilization), 1.0), 0.0)
    r = gpu_spec.calibration_factor
    p_idle = gpu_spec.idle_power_w
    p_max = gpu_spec.max_power_w

    if u <= 0.0:
        return float(p_idle)

    return float(p_idle + (p_max - p_idle) * (2.0 * u - u**r))


def _calculate_memory_utilization(used_gbs: float, peak_gbs: float) -> float:
    return min(1.0, used_gbs / peak_gbs)


def _estimate_memory_bandwidth_usage(
    model_params: float,
    batch_size: int,
    seq_length: int,
    step_time_s: float,
    hidden_dim: int = 4096,
) -> float:
    """Estimate per-step memory-bandwidth use (GB/s).

    Bytes use 1e9 (GB, not GiB) to match the GB-denominated capacity in
    simulate_training_step (gpu.bandwidth_bps / 1e9); GiB understated the
    reported gpu_memory_utilization by ~7%.
    """
    bytes_per_param = 2
    param_traffic = model_params * bytes_per_param * 5
    activation_traffic = batch_size * seq_length * hidden_dim * bytes_per_param
    return (param_traffic + activation_traffic) / 1e9 / step_time_s


def _lora_trainable_params(
    hidden_size: int,
    num_layers: int,
    rank: int = 8,
    target_modules: int = 4,
) -> int:
    return 2 * rank * hidden_size * target_modules * num_layers


def _ring_allreduce_time(
    gradient_bytes: float,
    num_participants: int,
    bandwidth_gbps: float,
    latency_s: float = 5e-6,
    overhead_per_msg_s: float = 2e-6,
) -> float:
    if num_participants <= 1:
        return 0.0
    bw = bandwidth_gbps * 1e9 / 8
    chunk = gradient_bytes * (num_participants - 1) / num_participants
    return latency_s * math.log2(num_participants) + overhead_per_msg_s * (num_participants - 1) + chunk / bw


def _comm_time(
    trainable_params: int,
    num_gpus: int,
    network_bandwidth_gbps: float,
    num_nodes: int = 1,
    calibrated: bool = True,
) -> float:
    if num_gpus <= 1:
        return 0.0
    grad_bytes = trainable_params * 4
    comm_scale = get_comm_scale() if calibrated else 1.0
    if num_nodes <= 1:
        return _ring_allreduce_time(grad_bytes, num_gpus, network_bandwidth_gbps) * comm_scale
    gpus_per_node = max(1, num_gpus // num_nodes)
    intra = _ring_allreduce_time(grad_bytes, gpus_per_node, network_bandwidth_gbps)
    inter = _ring_allreduce_time(grad_bytes, num_nodes, INFINIBAND_GBPS)
    return (intra + inter) * comm_scale


def simulate_training_step(
    model_name: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    method: str,
    num_gpus: int = 1,
    num_nodes: int = 1,
    grad_accum_steps: int = 1,
    backward_factor: float = 2.0,
    calibrated: bool = True,
) -> Dict[str, float]:
    """Simulate one optimizer step for a training config and return per-step metrics.

    Args:
        model_name, gpu_model: keys into the LLM / GPU spec libraries.
        tokens_per_sample, batch_size: sequence length and per-device micro-batch.
        method: "full", "lora" or "gptq-lora" (selects trainable-param count).
        num_gpus, num_nodes: total data-parallel width and node count.
        grad_accum_steps, backward_factor: micro-steps per update; bwd/fwd cost ratio.
        calibrated: apply fitted scales/corrections when True, else raw physics.

    Returns:
        dict with step_time_ms, tokens_per_second, tokens_per_step,
        gpu_compute_utilization (%), gpu_memory_utilization (%), gpu_power_watts.
    """
    llm = get_llm(model_name)
    gpu = get_gpu(gpu_model)

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    if backward_factor <= 0.0:
        raise ValueError(f"backward_factor must be > 0, got {backward_factor}")

    total_tokens = batch_size * tokens_per_sample
    flops = 2.0 * llm.active_params * total_tokens
    mfu = _compute_mfu(batch_size, gpu, calibrated)
    achieved_flops = gpu.fp_16_tensor_core_tflops * 1e12 * mfu
    overhead = get_training_overhead_s() if calibrated else 0.0
    forward_time = flops / achieved_flops + overhead

    backward_time = backward_factor * forward_time
    micro_step_time = forward_time + backward_time  # one fwd+bwd micro-step

    if method in ("lora", "gptq-lora"):
        trainable = _lora_trainable_params(llm.d_model, llm.n_layers)
    else:
        trainable = int(llm.m_params)
    optimizer_time = trainable * 20 / gpu.bandwidth_bps

    comm_time = _comm_time(trainable, num_gpus, gpu.network_bandwidth_gbps, num_nodes, calibrated)

    # One optimizer step accumulates `grad_accum_steps` micro-steps (fwd+bwd each),
    # then ONE optimizer update + all-reduce. Comm/optimizer are amortized over G.
    step_time_s = grad_accum_steps * micro_step_time + optimizer_time + comm_time

    mgc = get_multi_gpu_correction(num_gpus) if calibrated else 1.0
    throughput_scale = (
        (
            get_method_scale(method)
            * get_model_scale(model_name)
            * get_interaction_scale(model_name, method, gpu_model, num_gpus)
        )
        if calibrated
        else 1.0
    )
    # Data-parallel: each GPU runs its own batch, so per-step tokens scale with TOTAL num_gpus.
    tokens_per_step = grad_accum_steps * (batch_size * tokens_per_sample * num_gpus / mgc)
    tokens_per_second = tokens_per_step / step_time_s * throughput_scale

    bw_used = _estimate_memory_bandwidth_usage(
        llm.m_params,
        batch_size,
        tokens_per_sample,
        step_time_s,
        hidden_dim=llm.d_model,
    )
    memory_util = _calculate_memory_utilization(bw_used, gpu.bandwidth_bps / 1e9)
    power = _calculate_gpu_power(mfu, memory_util, gpu)

    return {
        "step_time_ms": step_time_s * 1000,
        "tokens_per_second": tokens_per_second,
        "tokens_per_step": tokens_per_step,
        "gpu_compute_utilization": mfu * 100,
        "gpu_memory_utilization": memory_util * 100,
        "gpu_power_watts": power,
    }


def _resolve_total_tokens(
    total_tokens: int | None,
    epochs: float | None,
    dataset_tokens: int | None,
) -> int | None:
    """Resolve a training job's token count from either ``total_tokens`` directly,
    or ``epochs`` + ``dataset_tokens`` (total = round(epochs * dataset_tokens); one
    epoch = one pass over the dataset). ``total_tokens`` wins if both are given;
    returns ``None`` when nothing is supplied (runtime then reported as 0)."""
    if total_tokens is not None:
        return total_tokens
    if epochs is None and dataset_tokens is None:
        return None
    if epochs is None or dataset_tokens is None:
        raise ValueError("pass epochs and dataset_tokens together (or use total_tokens)")
    if epochs < 0 or dataset_tokens < 0:
        raise ValueError("epochs and dataset_tokens must be non-negative")
    return int(round(epochs * dataset_tokens))


def simulate_full_training(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    number_gpus: int,
    number_nodes: int,
    total_tokens: int | None = None,
    epochs: float | None = None,
    dataset_tokens: int | None = None,
    grad_accum_steps: int = 1,
    backward_factor: float = 2.0,
) -> Dict[str, Any]:
    """Run a full training simulation (one step extrapolated over the whole job).

    Wraps ``simulate_training_step`` with total_gpus = number_gpus * number_nodes
    and derives job-level aggregates. Job size sets train_runtime (s): give
    ``total_tokens`` directly, or ``epochs`` + ``dataset_tokens``; otherwise 0.0.

    Returns:
        dict with train_tokens_per_second, train_tokens_per_gpu_per_second,
        train_samples_per_second, train_steps_per_second, train_runtime (s), plus
        the echoed config (model_name, gpu_name, method, batch_size,
        tokens_per_sample, number_gpus=total_gpus).
    """
    total_tokens = _resolve_total_tokens(total_tokens, epochs, dataset_tokens)
    total_gpus = number_gpus * number_nodes
    step = simulate_training_step(
        model_name,
        gpu_model,
        tokens_per_sample,
        batch_size,
        method,
        num_gpus=total_gpus,
        num_nodes=number_nodes,
        grad_accum_steps=grad_accum_steps,
        backward_factor=backward_factor,
    )
    tps = step["tokens_per_second"]
    # Reuse the engine's own per-step token count so steps/s is the exact inverse of tps.
    tokens_per_step = step["tokens_per_step"]
    return {
        "train_tokens_per_second": tps,
        # Divide by TOTAL gpus (gpus/node * nodes), not just gpus/node.
        "train_tokens_per_gpu_per_second": tps / total_gpus,
        "train_samples_per_second": tps / tokens_per_sample,
        "train_steps_per_second": tps / tokens_per_step,
        "train_runtime": total_tokens / tps if total_tokens is not None else 0.0,
        "model_name": model_name,
        "gpu_name": gpu_model,
        "method": method,
        "batch_size": batch_size,
        "tokens_per_sample": tokens_per_sample,
        "number_gpus": total_gpus,
    }
