"""Analytical training-step engine: FLOPs/MFU + comm/optimizer model -> throughput, runtime, GPU util and power."""

from __future__ import annotations

import math
from typing import Any, Dict

from kavier.sdk.energy.engine import mse_power
from kavier.sdk.io.constants import FLOPS_PER_PARAM_PER_TOKEN
from kavier.sdk.library.lookup import get_gpu, get_llm
from kavier.sdk.library.specs.GPUSpec import GPUSpec
from kavier.sdk.library.specs.LLMSpec import LLMSpec
from kavier.sdk.training.calibration import (
    get_comm_scale,
    get_interaction_scale,
    get_method_scale,
    get_mfu_batch_scale,
    get_mfu_multiplier,
    get_model_scale,
    get_multi_gpu_correction,
    get_training_overhead_s,
)
from kavier.sdk.training.core.config import (
    BITS_PER_BYTE,
    INFINIBAND_GBPS,
    RING_ALLREDUCE_LATENCY_S,
    RING_ALLREDUCE_OVERHEAD_PER_MSG_S,
    Method,
)
from kavier.sdk.units import FLOPS_PER_TFLOP, MS_PER_SECOND

# Adam moves ~20 bytes/param per optimizer step (fp32 weight + grad + 2 moments ~16 B, plus working copies).
_OPTIMIZER_BYTES_PER_PARAM = 20
# A step touches each weight ~5x in memory (weight read, grad read/write, 2 moments read/write).
_MEMORY_PASSES_PER_STEP = 5
# fp32 gradient width (bytes) sizing the all-reduce payload.
_FP32_BYTES = 4
# LoRA adapter shape: a rank-r update on `target_modules` projections per layer.
_LORA_RANK = 8
_LORA_TARGET_MODULES = 4


def _compute_mfu(batch_size: int, gpu: GPUSpec, calibrated: bool = True) -> float:
    base = gpu.mfu_factor * (get_mfu_multiplier(gpu.name) if calibrated else 1.0)
    alpha, beta = get_mfu_batch_scale()
    batch_scale = min(1.0, alpha * math.log2(batch_size) + beta)
    return float(base * batch_scale)


def _calculate_memory_utilization(used_gbs: float, peak_gbs: float) -> float:
    return min(1.0, used_gbs / peak_gbs)


def _estimate_memory_bandwidth_usage(
    model_params: float,
    batch_size: int,
    seq_length: int,
    step_time_s: float,
    hidden_dim: int,
    bytes_per_param: int,
) -> float:
    # Bytes use 1e9 (GB, not GiB) to match bandwidth_bps/1e9 in simulate_training_step; GiB understated util ~7%.
    param_traffic = model_params * bytes_per_param * _MEMORY_PASSES_PER_STEP
    activation_traffic = batch_size * seq_length * hidden_dim * bytes_per_param
    return (param_traffic + activation_traffic) / 1e9 / step_time_s


def _lora_trainable_params(
    hidden_size: int,
    num_layers: int,
    rank: int = _LORA_RANK,
    target_modules: int = _LORA_TARGET_MODULES,
) -> int:
    return 2 * rank * hidden_size * target_modules * num_layers


def _ring_allreduce_time(
    gradient_bytes: float,
    num_participants: int,
    bandwidth_gbps: float,
    latency_s: float = RING_ALLREDUCE_LATENCY_S,
    overhead_per_msg_s: float = RING_ALLREDUCE_OVERHEAD_PER_MSG_S,
) -> float:
    if num_participants <= 1:
        return 0.0
    bw: float = bandwidth_gbps * 1e9 / BITS_PER_BYTE
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
    grad_bytes = trainable_params * _FP32_BYTES
    comm_scale = get_comm_scale() if calibrated else 1.0
    if num_nodes <= 1:
        return _ring_allreduce_time(grad_bytes, num_gpus, network_bandwidth_gbps) * comm_scale
    gpus_per_node = max(1, num_gpus // num_nodes)
    intra = _ring_allreduce_time(grad_bytes, gpus_per_node, network_bandwidth_gbps)
    inter = _ring_allreduce_time(grad_bytes, num_nodes, INFINIBAND_GBPS)
    return (intra + inter) * comm_scale


def _validate_step_args(
    batch_size: int,
    grad_accum_steps: int,
    backward_factor: float,
    tokens_per_sample: int,
    num_gpus: int,
) -> None:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    if backward_factor <= 0.0:
        raise ValueError(f"backward_factor must be > 0, got {backward_factor}")
    if tokens_per_sample < 1:
        raise ValueError(f"tokens_per_sample must be >= 1, got {tokens_per_sample}")
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")


def _micro_step_time(
    llm: LLMSpec,
    gpu: GPUSpec,
    batch_size: int,
    tokens_per_sample: int,
    backward_factor: float,
    calibrated: bool,
) -> tuple[float, float]:
    """Forward+backward time for one micro-step, plus the physical MFU used for power. Returns (time_s, mfu)."""
    total_tokens = batch_size * tokens_per_sample
    # Compute uses ACTIVE params (MoE runs only active experts); optimizer/comm/memory below use total m_params.
    flops = FLOPS_PER_PARAM_PER_TOKEN * llm.active_params * total_tokens
    mfu = _compute_mfu(batch_size, gpu, calibrated)
    achieved_flops = gpu.fp_16_tensor_core_tflops * FLOPS_PER_TFLOP * mfu
    overhead = get_training_overhead_s() if calibrated else 0.0
    forward_time = flops / achieved_flops + overhead

    # backward ~2x forward FLOPs (standard rule of thumb; override via backward_factor).
    backward_time = backward_factor * forward_time
    return forward_time + backward_time, mfu


def _trainable_params(llm: LLMSpec, method: str) -> int:
    if method in (Method.LORA, Method.GPTQ_LORA):
        return _lora_trainable_params(llm.d_model, llm.n_layers)
    return int(llm.m_params)


def _throughput_scale(model_name: str, method: str, gpu_model: str, num_gpus: int, calibrated: bool) -> float:
    if not calibrated:
        return 1.0
    return (
        get_method_scale(method)
        * get_model_scale(model_name)
        * get_interaction_scale(model_name, method, gpu_model, num_gpus)
    )


def _step_result(
    step_time_s: float,
    tokens_per_second: float,
    tokens_per_step: float,
    mfu: float,
    memory_util: float,
    power: float,
) -> Dict[str, float]:
    return {
        "step_time_ms": step_time_s * MS_PER_SECOND,
        "tokens_per_second": tokens_per_second,
        "tokens_per_step": tokens_per_step,
        "gpu_compute_utilization": mfu * 100,  # raw physical MFU, not throughput_scale-adjusted
        "gpu_memory_utilization": memory_util * 100,
        "gpu_power_watts": power,
    }


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
    """Simulate one optimizer step; ``calibrated`` applies fitted scales/corrections, else raw physics. Util in %."""
    llm = get_llm(model_name)
    gpu = get_gpu(gpu_model)

    _validate_step_args(batch_size, grad_accum_steps, backward_factor, tokens_per_sample, num_gpus)

    micro_step_time, mfu = _micro_step_time(llm, gpu, batch_size, tokens_per_sample, backward_factor, calibrated)

    trainable = _trainable_params(llm, method)
    optimizer_time = trainable * _OPTIMIZER_BYTES_PER_PARAM / gpu.bandwidth_bps

    comm_time = _comm_time(trainable, num_gpus, gpu.network_bandwidth_gbps, num_nodes, calibrated)

    # One step = grad_accum_steps micro-steps + ONE optimizer update + all-reduce.
    step_time_s = grad_accum_steps * micro_step_time + optimizer_time + comm_time

    mgc = get_multi_gpu_correction(num_gpus) if calibrated else 1.0
    throughput_scale = _throughput_scale(model_name, method, gpu_model, num_gpus, calibrated)
    # Data-parallel: per-step tokens scale with TOTAL num_gpus.
    tokens_per_step = grad_accum_steps * (batch_size * tokens_per_sample * num_gpus / mgc)
    tokens_per_second = tokens_per_step / step_time_s * throughput_scale

    bw_used = _estimate_memory_bandwidth_usage(
        llm.m_params,
        batch_size,
        tokens_per_sample,
        step_time_s,
        hidden_dim=llm.d_model,
        bytes_per_param=llm.p_bytes,
    )
    memory_util = _calculate_memory_utilization(bw_used, gpu.bandwidth_bps / 1e9)
    power = mse_power(mfu, memory_util, gpu)

    return _step_result(step_time_s, tokens_per_second, tokens_per_step, mfu, memory_util, power)


def _resolve_total_tokens(
    total_tokens: int | None,
    epochs: float | None,
    dataset_tokens: int | None,
) -> int | None:
    """``total_tokens`` (wins) or round(epochs * dataset_tokens); ``None`` if neither."""
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
    """One step extrapolated over the whole job (total_gpus = number_gpus * number_nodes); job size sets runtime(s)."""
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
    tokens_per_step = step["tokens_per_step"]
    return {
        "train_tokens_per_second": tps,
        # Divide by TOTAL gpus (gpus/node * nodes), not just gpus/node.
        "train_tokens_per_gpu_per_second": tps / total_gpus,
        "train_samples_per_second": tps / tokens_per_sample,
        "train_steps_per_second": tps / tokens_per_step,
        "train_runtime": total_tokens / tps if total_tokens is not None else 0.0,
        "total_tokens": total_tokens,
        "model_name": model_name,
        "gpu_name": gpu_model,
        "method": method,
        "batch_size": batch_size,
        "tokens_per_sample": tokens_per_sample,
        "number_gpus": total_gpus,
    }
