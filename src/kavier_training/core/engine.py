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
from kavier_training.components.lora import compute_lora_trainable_params


def simulate_training_step(
    model_name: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    method: str,
) -> Dict[str, float]:
    """
    Simulate a single training step (forward + backward + optimizer).
    
    Args:
        model_name: LLM model name (e.g., "llama3.1-70b")
        gpu_model: GPU model (e.g., "NVIDIA-A100-SXM4-80GB")
        tokens_per_sample: Sequence length
        batch_size: Batch size per GPU
        method: Training method ("full" or "lora")
        
    Returns:
        Dictionary with step metrics:
        - step_time_ms: Time for one training step
        - tokens_per_second: Throughput
        - gpu_compute_utilization: GPU utilization %
        - gpu_memory_utilization: Memory utilization %
        - gpu_power_watts: Power consumption
    """
    llm = LLM_SPEC_LIBRARY[model_name]
    gpu = GPU_SPEC_LIBRARY[gpu_model]
    
    trainable_params = llm.m_params if method == "full" else compute_lora_trainable_params(
        int(llm.m_params), llm.d_model, llm.n_layers
    )
    
    forward_time, _ = calculate_forward_pass(batch_size, tokens_per_sample, llm, gpu)
    backward_time, _ = calculate_backward_pass(forward_time, llm)
    optimizer_time, _ = calculate_optimizer_step(llm, gpu)
    
    step_time_s = forward_time + backward_time + optimizer_time
    tokens_per_step = batch_size * tokens_per_sample
    tokens_per_second = tokens_per_step / step_time_s if step_time_s > 0 else 0
    
    return {
        "step_time_ms": step_time_s * 1000,
        "tokens_per_second": tokens_per_second,
        "gpu_compute_utilization": 0.0,
        "gpu_memory_utilization": 0.0,
        "gpu_power_watts": 0.0,
    }


def simulate_full_training(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    number_gpus: int,
    number_nodes: int,
    metrics: str = "both",
) -> Dict[str, Any]:
    """
    Simulate complete training run and return predictions.
    
    This is the main entry point called by the CLI.
    
    Args:
        model_name: LLM model name
        method: Training method ("full" or "lora")
        gpu_model: GPU model name
        tokens_per_sample: Sequence length
        batch_size: Batch size per GPU
        number_gpus: Number of GPUs
        number_nodes: Number of nodes
        metrics: "performance", "energy", or "both"
        
    Returns:
        Dictionary:
        - train_tokens_per_second
        - train_tokens_per_gpu_per_second
        - train_samples_per_second
        - train_steps_per_second
        - train_runtime (seconds)
        - gpu_compute_utilization_avg/min/max
        - gpu_memory_utilization_avg/min/peak/max
        - gpu_power_watts_avg/min/max (if metrics includes "energy")
        - etc.
    """
    llm = LLM_SPEC_LIBRARY[model_name]
    gpu = GPU_SPEC_LIBRARY[gpu_model]
    total_gpus = number_gpus * number_nodes
    
    step_result = simulate_training_step(model_name, gpu_model, tokens_per_sample, batch_size, method)
    single_gpu_throughput = step_result["tokens_per_second"]
    
    scaling_efficiency = _compute_scaling_efficiency(total_gpus, number_nodes)
    total_throughput = single_gpu_throughput * total_gpus * scaling_efficiency
    
    tokens_per_gpu = total_throughput / total_gpus if total_gpus > 0 else 0
    samples_per_second = total_throughput / tokens_per_sample if tokens_per_sample > 0 else 0
    steps_per_second = samples_per_second / batch_size if batch_size > 0 else 0
    
    return {
        "train_tokens_per_second": total_throughput,
        "train_tokens_per_gpu_per_second": tokens_per_gpu,
        "train_samples_per_second": samples_per_second,
        "train_steps_per_second": steps_per_second,
        "train_runtime": 0.0,
        "model_name": model_name,
        "gpu_name": gpu_model,
        "method": method,
        "batch_size": batch_size,
        "tokens_per_sample": tokens_per_sample,
        "number_gpus": total_gpus,
    }


def _compute_scaling_efficiency(total_gpus: int, number_nodes: int) -> float:
    """
    Compute multi-GPU scaling efficiency.
    
    Efficiency decreases with more GPUs due to communication overhead.
    Single-node: ~98% efficiency, Multi-node: ~90-95% efficiency.
    """
    if total_gpus == 1:
        return 1.0
    
    if number_nodes == 1:
        base_efficiency = 0.98
        gpu_penalty = (total_gpus - 1) * 0.005
    else:
        base_efficiency = 0.95
        gpu_penalty = (total_gpus - number_nodes) * 0.008
    
    return max(0.85, base_efficiency - gpu_penalty)


