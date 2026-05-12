"""
Kavier Training Simulator — analytical throughput model for LLM fine-tuning.

Predicts tokens-per-second from first principles using a single closed-form
equation (see ``simulate_training_step`` docstring for the full formula).

The step time decomposes into four physics-based components:

    T_step = T_forward + T_backward + T_optimizer + T_comm

Each term is derived from peer-reviewed literature:

    [1] Shoeybi et al. 2019.  "Megatron-LM: Training Multi-Billion Parameter
        Language Models Using Model Parallelism."  arXiv:1909.08053.
        — 2·P·T FLOPs per forward pass; backward ≈ 2× forward.

    [2] Chowdhery et al. 2022.  "PaLM: Scaling Language Modeling with Pathways."
        JMLR 24(240):1–113.
        — Definition of Model FLOPs Utilization (MFU).

    [3] Williams et al. 2009.  "Roofline: An Insightful Visual Performance Model
        for Multicore Architectures."  Communications of the ACM 52(4):65–76.
        — Batch-size MFU scaling via arithmetic-intensity / roofline analysis.

    [4] Rajbhandari et al. 2020.  "ZeRO: Memory Optimizations Toward Training
        Trillion Parameter Models."  SC '20.
        — AdamW optimizer memory: 20 bytes/param transfer per step.

    [5] Thakur et al. 2005.  "Optimization of Collective Communication
        Operations in MPI."  IJHPCA 19(1):49–66.
        — Ring all-reduce communication model.

    [6] Hu et al. 2022.  "LoRA: Low-Rank Adaptation of Large Language Models."
        ICLR 2022.
        — LoRA trainable parameters: 2·r·d per adapter module.

    [7] Narayanan et al. 2021.  "Efficient Large-Scale Language Model Training
        on GPU Clusters Using Megatron-LM."  SC '21.
        — Backward = 2× forward; hierarchical communication.

    [8] Fedus et al. 2022.  "Switch Transformers: Scaling to Trillion Parameter
        Models with Simple and Efficient Sparsity."  JMLR 23(120):1–39.
        — MoE routing overhead (~1–2 % of forward time).

    [9] Loshchilov & Hutter 2019.  "Decoupled Weight Decay Regularization."
        ICLR 2019.
        — AdamW optimizer (momentum + variance states in fp32).
"""

from __future__ import annotations

import math
from typing import Dict, Any

from library.llm import LLM_SPEC_LIBRARY
from library.gpu import GPU_SPEC_LIBRARY
from library.specs.GPUSpec import GPUSpec
from library.specs.LLMSpec import LLMSpec
from kavier_training.core.config import get_training_compute_efficiency
from kavier_training.core.calibration import (
    get_multi_gpu_correction, get_method_scale, get_model_scale,
    get_comm_scale, get_training_overhead_s,
)
from kavier_training.components.energy import (
    calculate_gpu_power, calculate_memory_utilization,
    estimate_memory_bandwidth_usage,
)


# ---------------------------------------------------------------------------
# Communication constants
# ---------------------------------------------------------------------------
INFINIBAND_GBPS = 200.0  # inter-node bandwidth (HDR InfiniBand)


# ---------------------------------------------------------------------------
# Core physics helpers
# ---------------------------------------------------------------------------

def _lora_trainable_params(
    hidden_size: int, num_layers: int,
    rank: int = 8, target_modules: int = 4,
) -> int:
    """LoRA adapter parameter count [6]."""
    return 2 * rank * hidden_size * target_modules * num_layers


def _ring_allreduce_time(
    gradient_bytes: float,
    num_participants: int,
    bandwidth_gbps: float,
    latency_s: float = 5e-6,
    overhead_per_msg_s: float = 2e-6,
) -> float:
    """Ring all-reduce time [5]."""
    if num_participants <= 1:
        return 0.0
    bw = bandwidth_gbps * 1e9 / 8  # bytes/s
    chunk = gradient_bytes * (num_participants - 1) / num_participants
    return (
        latency_s * math.log2(num_participants)
        + overhead_per_msg_s * (num_participants - 1)
        + chunk / bw
    )


def _comm_time(
    trainable_params: int,
    num_gpus: int,
    network_bandwidth_gbps: float,
    num_nodes: int = 1,
) -> float:
    """Hierarchical all-reduce: intra-node (NVLink) + inter-node (IB) [5, 7]."""
    if num_gpus <= 1:
        return 0.0
    grad_bytes = trainable_params * 4  # fp32 gradients
    if num_nodes <= 1:
        return _ring_allreduce_time(grad_bytes, num_gpus, network_bandwidth_gbps) * get_comm_scale()
    gpus_per_node = max(1, num_gpus // num_nodes)
    intra = _ring_allreduce_time(grad_bytes, gpus_per_node, network_bandwidth_gbps)
    inter = _ring_allreduce_time(grad_bytes, num_nodes, INFINIBAND_GBPS)
    return (intra + inter) * get_comm_scale()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_training_step(
    model_name: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    method: str,
    num_gpus: int = 1,
    num_nodes: int = 1,
    multi_gpu_correction: float | None = None,
    fms_version: str | None = None,
    torch_dtype: str | None = None,
) -> Dict[str, float]:
    r"""
    Predict throughput for one training step using the analytical model.

    .. math::

        \text{TPS} = \frac{B \cdot S \cdot G_\text{node}
                           \cdot \sigma_\text{method} \cdot \sigma_\text{model}}
                          {\text{MGC} \cdot T_\text{step}}

    where

    .. math::

        T_\text{step} = \underbrace{\frac{6\,P_\text{active}\,B\,S}
                                    {F_\text{peak}\,\text{MFU}}}_{
                          \text{forward + backward [1,7]}}
                       + 3\,T_\text{oh}
                       + \underbrace{\frac{20\,P_\text{train}}{BW_\text{mem}}}_{
                          \text{optimizer [4,9]}}
                       + \underbrace{T_\text{comm}}_{\text{all-reduce [5]}}

    Parameters
    ----------
    model_name : str
        Key into ``LLM_SPEC_LIBRARY``.
    gpu_model : str
        Key into ``GPU_SPEC_LIBRARY``.
    tokens_per_sample : int
        Sequence length *S*.
    batch_size : int
        Per-GPU batch size *B*.
    method : str
        ``"full"``, ``"lora"``, or ``"gptq-lora"``.
    num_gpus : int
        Total GPUs across all nodes.
    num_nodes : int
        Physical node count.

    Returns
    -------
    dict
        ``step_time_ms``, ``tokens_per_second``,
        ``gpu_compute_utilization``, ``gpu_memory_utilization``,
        ``gpu_power_watts``.
    """
    llm = LLM_SPEC_LIBRARY[model_name]
    gpu = GPU_SPEC_LIBRARY[gpu_model]

    # --- Forward pass [1]: FLOPs = 2·P_active·B·S -------------------------
    total_tokens = batch_size * tokens_per_sample
    flops = 2.0 * llm.active_params * total_tokens
    mfu = get_training_compute_efficiency(batch_size, tokens_per_sample, gpu)
    achieved_flops = gpu.fp_16_tensor_core_tflops * 1e12 * mfu
    forward_time = flops / achieved_flops + get_training_overhead_s()
    if llm.is_moe:
        forward_time *= 1.015  # ~1.5 % routing overhead [8]

    # --- Backward pass [1, 7]: T_bwd = 2·T_fwd ----------------------------
    backward_time = 2.0 * forward_time

    # --- Trainable parameters & optimizer [4, 6, 9] -----------------------
    if method in ("lora", "gptq-lora"):
        trainable = _lora_trainable_params(llm.d_model, llm.n_layers)
    else:
        trainable = int(llm.m_params)
    optimizer_time = trainable * 20 / gpu.bandwidth_bps  # 20 bytes/param [4]

    # --- Communication [5, 7] ---------------------------------------------
    comm_time = _comm_time(
        trainable, num_gpus, gpu.network_bandwidth_gbps, num_nodes,
    )

    # --- Assemble step time ------------------------------------------------
    step_time_s = forward_time + backward_time + optimizer_time + comm_time

    # --- Throughput --------------------------------------------------------
    if multi_gpu_correction is None:
        multi_gpu_correction = get_multi_gpu_correction(num_gpus)
    throughput_scale = get_method_scale(method) * get_model_scale(model_name)
    gpus_per_node = max(1, num_gpus // num_nodes) if num_nodes > 0 else num_gpus
    tokens_per_step = batch_size * tokens_per_sample * gpus_per_node / multi_gpu_correction
    tokens_per_second = (tokens_per_step / step_time_s * throughput_scale) if step_time_s > 0 else 0

    # --- Energy metrics (secondary) ----------------------------------------
    compute_util = mfu
    bw_used = estimate_memory_bandwidth_usage(
        llm.m_params, batch_size, tokens_per_sample, step_time_s,
        hidden_dim=llm.d_model,
    )
    memory_util = calculate_memory_utilization(bw_used, gpu.bandwidth_bps / 1e9)
    power = calculate_gpu_power(compute_util, memory_util, gpu)

    return {
        "step_time_ms": step_time_s * 1000,
        "tokens_per_second": tokens_per_second,
        "gpu_compute_utilization": compute_util * 100,
        "gpu_memory_utilization": memory_util * 100,
        "gpu_power_watts": power,
    }


def simulate_full_training(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    number_gpus: int,
    number_nodes: int,
    total_tokens: int | None = None,
    metrics: str = "both",
) -> Dict[str, Any]:
    """Simulate a complete training run (entry point for the CLI)."""
    total_gpus = number_gpus * number_nodes
    step = simulate_training_step(
        model_name, gpu_model, tokens_per_sample, batch_size, method,
        num_gpus=total_gpus, num_nodes=number_nodes,
    )
    tps = step["tokens_per_second"]
    gpus_per_node = max(1, number_gpus)
    return {
        "train_tokens_per_second": tps,
        "train_tokens_per_gpu_per_second": tps / gpus_per_node if gpus_per_node else tps,
        "train_samples_per_second": tps / tokens_per_sample if tokens_per_sample else 0,
        "train_steps_per_second": tps / (tokens_per_sample * batch_size) if tokens_per_sample * batch_size else 0,
        "train_runtime": (total_tokens / tps) if (total_tokens and tps > 0) else 0.0,
        "model_name": model_name,
        "gpu_name": gpu_model,
        "method": method,
        "batch_size": batch_size,
        "tokens_per_sample": tokens_per_sample,
        "number_gpus": total_gpus,
    }
