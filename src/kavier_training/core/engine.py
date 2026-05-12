"""
Main training simulation module for Kavier.

This module simulates LLM fine-tuning workloads and predicts:
- Performance: throughput (tokens/sec), runtime
- Energy: GPU power consumption
- Utilization: GPU compute and memory utilization
"""

from __future__ import annotations

from typing import Dict, Any

from library.llm import LLM_SPEC_LIBRARY
from library.gpu import GPU_SPEC_LIBRARY
from kavier_training.components.forward_pass import calculate_forward_pass
from kavier_training.components.backward_pass import calculate_backward_pass
from kavier_training.components.optimizer import calculate_optimizer_step
from kavier_training.components.lora import (
    compute_lora_trainable_params,
    calculate_lora_backward_pass,
    calculate_lora_optimizer_step,
)
from kavier_training.components.communication import simulate_allreduce
from kavier_training.components.energy import (
    calculate_gpu_power,
    calculate_compute_utilization,
    calculate_memory_utilization,
    estimate_memory_bandwidth_usage,
)
from kavier_training.core.config import get_training_compute_efficiency
from kavier_training.core.calibration import (
    get_multi_gpu_correction, get_method_scale, get_model_scale,
    get_version_scale, get_dtype_scale, get_model_method_scale,
    get_batch_size_correction, get_model_method_version_scale,
    get_model_method_gpucount_scale,
)


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
    """
    Simulate a single training step (forward + backward + optimizer).
    
    Args:
        model_name: LLM model name (e.g., "llama3.1-70b")
        gpu_model: GPU model (e.g., "NVIDIA-A100-SXM4-80GB")
        tokens_per_sample: Sequence length
        batch_size: Batch size per GPU
        method: Training method ("full" or "lora")
        num_gpus: Total number of GPUs across all nodes.
        num_nodes: Number of physical nodes (default 1).
        
    Returns:
        Dictionary with step metrics:
        - step_time_ms: Time for one training step
        - tokens_per_second: Throughput (matches dataset_tokens_per_second,
          which reports per-node throughput for multi-node jobs)
        - gpu_compute_utilization: GPU utilization %
        - gpu_memory_utilization: Memory utilization %
        - gpu_power_watts: Power consumption
    """
    llm = LLM_SPEC_LIBRARY[model_name]
    gpu = GPU_SPEC_LIBRARY[gpu_model]
    
    # Calculate forward pass (same for full and LoRA)
    forward_time, _ = calculate_forward_pass(batch_size, tokens_per_sample, llm, gpu)
    
    # Calculate backward and optimizer based on method
    if method == "lora" or method == "gptq-lora":
        trainable_params = compute_lora_trainable_params(
            int(llm.m_params), llm.d_model, llm.n_layers
        )
        backward_time, _ = calculate_lora_backward_pass(
            forward_time, trainable_params, int(llm.m_params)
        )
        optimizer_time, _ = calculate_lora_optimizer_step(
            trainable_params, gpu.bandwidth_bps
        )
    else:  # full fine-tuning
        trainable_params = llm.m_params
        backward_time, _ = calculate_backward_pass(forward_time, llm)
        optimizer_time, _ = calculate_optimizer_step(llm, gpu)
    
    # Communication: hierarchical allreduce (intra-node NVLink, inter-node IB)
    comm_time = simulate_allreduce(
        int(trainable_params),
        num_gpus,
        gpu.network_bandwidth_gbps,
        num_nodes=num_nodes,
    ) if num_gpus > 1 else 0.0
    
    step_time_s = forward_time + backward_time + optimizer_time + comm_time
    
    if multi_gpu_correction is None:
        multi_gpu_correction = get_multi_gpu_correction(num_gpus)
    
    # Throughput scales capturing kernel/framework/software-version efficiency
    version_s = get_version_scale(fms_version) if fms_version else 1.0
    dtype_s = get_dtype_scale(torch_dtype) if torch_dtype else 1.0
    mm_s = get_model_method_scale(model_name, method)
    bs_s = get_batch_size_correction(batch_size)
    mmv_s = get_model_method_version_scale(model_name, method, fms_version) if fms_version else 1.0
    mmg_s = get_model_method_gpucount_scale(model_name, method, num_gpus)
    throughput_scale = get_method_scale(method) * get_model_scale(model_name) * version_s * dtype_s * mm_s * bs_s * mmv_s * mmg_s

    # dataset_tokens_per_second in the training data reports per-node throughput
    # for multi-node jobs (= per_gpu_tps * gpus_per_node).
    gpus_per_node = max(1, num_gpus // num_nodes) if num_nodes > 0 else num_gpus
    tokens_per_gpu = batch_size * tokens_per_sample
    total_tokens_per_step = tokens_per_gpu * gpus_per_node / multi_gpu_correction
    tokens_per_second = (total_tokens_per_step / step_time_s * throughput_scale) if step_time_s > 0 else 0
    
    # Calculate energy metrics
    # Compute utilization based on MFU
    mfu = get_training_compute_efficiency(batch_size, tokens_per_sample, gpu)
    compute_util = mfu  # MFU directly represents compute utilization
    
    # Memory bandwidth utilization
    bandwidth_used = estimate_memory_bandwidth_usage(
        llm.m_params, batch_size, tokens_per_sample, step_time_s, hidden_dim=llm.d_model
    )
    peak_bandwidth_gbs = gpu.bandwidth_bps / 1e9
    memory_util = calculate_memory_utilization(bandwidth_used, peak_bandwidth_gbs)
    
    # GPU power consumption
    power_watts = calculate_gpu_power(compute_util, memory_util, gpu)
    
    return {
        "step_time_ms": step_time_s * 1000,
        "tokens_per_second": tokens_per_second,
        "gpu_compute_utilization": compute_util * 100,  # Convert to percentage
        "gpu_memory_utilization": memory_util * 100,    # Convert to percentage
        "gpu_power_watts": power_watts,
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
    """
    Simulate complete training run and return predictions.
    
    This is the main entry point called by the CLI.
    """
    total_gpus = number_gpus * number_nodes
    
    step_result = simulate_training_step(
        model_name, gpu_model, tokens_per_sample, batch_size, method,
        num_gpus=total_gpus, num_nodes=number_nodes,
    )
    total_throughput = step_result["tokens_per_second"]
    
    gpus_per_node = max(1, number_gpus)
    tokens_per_gpu = total_throughput / gpus_per_node if gpus_per_node > 0 else total_throughput
    samples_per_second = total_throughput / tokens_per_sample if tokens_per_sample > 0 else 0
    steps_per_second = samples_per_second / batch_size if batch_size > 0 else 0
    
    train_runtime = (total_tokens / total_throughput) if (total_tokens and total_throughput > 0) else 0.0
    
    return {
        "train_tokens_per_second": total_throughput,
        "train_tokens_per_gpu_per_second": tokens_per_gpu,
        "train_samples_per_second": samples_per_second,
        "train_steps_per_second": steps_per_second,
        "train_runtime": train_runtime,
        "model_name": model_name,
        "gpu_name": gpu_model,
        "method": method,
        "batch_size": batch_size,
        "tokens_per_sample": tokens_per_sample,
        "number_gpus": total_gpus,
    }



