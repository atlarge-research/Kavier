"""
Training configuration library for Kavier.

Simplified library matching the ado-sfttrainer dataset structure.
"""

from typing import Dict, Any


def compute_lora_params(
    num_parameters: int,
    hidden_size: int,
    num_layers: int,
    lora_rank: int = 8,
) -> int:
    """
    Compute number of trainable parameters for LoRA.
    
    LoRA adds low-rank matrices to attention layers (Q, K, V, O projections).
    Each LoRA module adds: 2 * rank * hidden_size parameters.
    
    Args:
        num_parameters: Total model parameters
        hidden_size: Model hidden dimension
        num_layers: Number of transformer layers
        lora_rank: LoRA rank (r)
        
    Returns:
        Number of trainable LoRA parameters
    """
    # 4 target modules per layer: Q, K, V, O projections
    target_modules = 4
    params_per_module = 2 * lora_rank * hidden_size
    total_lora_params = params_per_module * target_modules * num_layers
    
    return total_lora_params


def estimate_memory_usage(
    num_parameters: int,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    method: str = "full",
) -> Dict[str, float]:
    """
    Estimate memory usage for training (simplified).
    
    Args:
        num_parameters: Total model parameters
        batch_size: Batch size per GPU
        sequence_length: Sequence length (tokens_per_sample)
        hidden_size: Model hidden dimension
        num_layers: Number of layers
        method: Training method (full, lora, qlora)
        
    Returns:
        Dictionary with memory breakdown in GB
    """
    # Assume bfloat16 (2 bytes per param)
    bytes_per_param = 2
    
    # Model weights
    model_memory_gb = (num_parameters * bytes_per_param) / (1024**3)
    
    # Determine trainable parameters based on method
    if method in ["lora", "qlora"]:
        # Only ~1% of params are trainable in LoRA
        trainable_params = num_parameters * 0.01
    else:
        trainable_params = num_parameters
    
    # Optimizer states (AdamW: 2x params for momentum + variance)
    optimizer_memory_gb = (trainable_params * 4 * 2.0) / (1024**3)
    
    # Gradients (FP32)
    gradient_memory_gb = (trainable_params * 4) / (1024**3)
    
    # Activations (simplified: batch_size * seq_len * hidden_size * num_layers)
    activation_memory_gb = (
        batch_size * sequence_length * hidden_size * num_layers * bytes_per_param
    ) / (1024**3)
    
    # Total
    total_memory_gb = (
        model_memory_gb + optimizer_memory_gb + gradient_memory_gb + activation_memory_gb
    )
    
    return {
        "model_gb": model_memory_gb,
        "optimizer_gb": optimizer_memory_gb,
        "gradients_gb": gradient_memory_gb,
        "activations_gb": activation_memory_gb,
        "total_gb": total_memory_gb,
    }


