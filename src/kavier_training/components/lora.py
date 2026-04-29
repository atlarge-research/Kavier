"""
LoRA (Low-Rank Adaptation) efficiency modeling.

Simulates the efficiency gains from using LoRA vs full fine-tuning.
"""

from __future__ import annotations

from typing import Dict, Any


def calculate_lora_backward_pass(
    forward_time_s: float,
    trainable_params: int,
    total_params: int,
) -> tuple[float, float]:
    """
    Calculate backward pass for LoRA training.
    
    LoRA backpropagates through full model (same compute as full fine-tuning),
    but only computes/stores gradients for adapter params (~0.1-1%).
    Time is dominated by backprop, not gradient computation.
    
    Args:
        forward_time_s: Forward pass time
        trainable_params: LoRA trainable parameters
        total_params: Total model parameters
        
    Returns:
        Tuple of (backward_time_s, gradient_memory_gb)
    """
    # LoRA backward: full backprop (2x forward), minimal gradient savings
    backward_time_s = 2.0 * forward_time_s
    
    # Gradient memory only for LoRA params (fp16)
    gradient_memory_bytes = trainable_params * 2  # 2 bytes for fp16
    gradient_memory_gb = gradient_memory_bytes / (1024**3)
    
    return backward_time_s, gradient_memory_gb


def calculate_lora_optimizer_step(
    trainable_params: int,
    gpu_bandwidth_bps: float,
) -> tuple[float, float]:
    """
    Calculate optimizer step for LoRA training.
    
    AdamW only updates LoRA parameters, not full model.
    
    Args:
        trainable_params: LoRA trainable parameters
        gpu_bandwidth_bps: GPU memory bandwidth
        
    Returns:
        Tuple of (optimizer_time_s, optimizer_memory_gb)
    """
    # Optimizer memory: 2 states (momentum + variance) in fp32
    optimizer_memory_bytes = 2 * trainable_params * 4
    optimizer_memory_gb = optimizer_memory_bytes / (1024**3)
    
    # Optimizer time: 20 bytes per param (same as full)
    bytes_per_param_transfer = 20
    total_bytes = trainable_params * bytes_per_param_transfer
    optimizer_time_s = total_bytes / gpu_bandwidth_bps
    
    return optimizer_time_s, optimizer_memory_gb


def compute_lora_trainable_params(
    total_params: int,
    hidden_size: int,
    num_layers: int,
    lora_rank: int = 8,
    target_modules: int = 4,
) -> int:
    """
    Compute number of trainable parameters for LoRA.
    
    LoRA adds low-rank matrices to attention layers.
    Each LoRA module: 2 * rank * hidden_size parameters
    
    Args:
        total_params: Total model parameters
        hidden_size: Model hidden dimension
        num_layers: Number of transformer layers
        lora_rank: LoRA rank
        target_modules: Number of modules per layer (Q, K, V, O = 4)
        
    Returns:
        Number of trainable LoRA parameters
    """
    # TODO: Use library.training_library.compute_lora_params
    params_per_module = 2 * lora_rank * hidden_size
    total_lora_params = params_per_module * target_modules * num_layers
    return total_lora_params


