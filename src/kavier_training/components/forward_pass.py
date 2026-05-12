"""
Forward pass simulation for training.

References:
- Vaswani et al. 2017: "Attention Is All You Need" - Transformer architecture
- Shoeybi et al. 2019: "Megatron-LM" - FLOPs analysis for transformer training
- Rajbhandari et al. 2020: "ZeRO" - Memory optimizations and activation memory
"""

from library.specs.GPUSpec import GPUSpec
from library.specs.LLMSpec import LLMSpec
from kavier_training.core.config import get_training_compute_efficiency
from kavier_training.core.calibration import get_training_overhead_s


def calculate_forward_pass(
    batch_size: int,
    seq_length: int,
    llm: LLMSpec,
    gpu: GPUSpec,
) -> tuple[float, float]:
    """
    Calculate forward pass time and activation memory for training.
    
    Forward pass computation is identical to inference prefill - processing
    input tokens through the transformer layers. The key difference is that
    training must store activations for the backward pass.
    
    Time Calculation:
    - Reuses get_prefill_time_s() which implements: T = overhead + (tokens x 2xparams) / GPU_FLOPS
    - Based on Shoeybi et al. 2019 (Megatron-LM): ~2 FLOPs per parameter per token
    
    Memory Calculation:
    - Based on Rajbhandari et al. 2020 (ZeRO paper)
    - Activation memory: batch_size x seq_length x hidden_dim x num_layers x bytes_per_element
    - Simplified model (excludes attention matrices for now)
    
    Args:
        batch_size: Training batch size
        seq_length: Sequence length (tokens per sample)
        llm: LLM specifications
        gpu: GPU specifications
        
    Returns:
        Tuple of (forward_time_s, activation_memory_gb)
        
    References:
        [1] Vaswani et al. 2017: "Attention Is All You Need"
        [2] Shoeybi et al. 2019: "Megatron-LM: Training Multi-Billion Parameter Language Models"
        [3] Rajbhandari et al. 2020: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
    """
    # Calculate forward pass time with training-specific MFU
    # Training has lower MFU than inference due to activation checkpointing
    total_tokens = batch_size * seq_length
    
    # FLOPs: 2 operations per parameter per token (Shoeybi et al. 2019)
    # For MoE models, use active_params (only active experts contribute FLOPs)
    flops_required = 2.0 * llm.active_params * total_tokens
    
    # Achieved FLOPS based on batch size, GPU specs, and architecture
    mfu = get_training_compute_efficiency(batch_size, seq_length, gpu)
    achieved_flops = gpu.fp_16_tensor_core_tflops * 1e12 * mfu
    
    # Time = FLOPs / achieved_FLOPS + overhead
    forward_time_s = (flops_required / achieved_flops) + get_training_overhead_s()
    
    # MoE routing overhead: expert selection adds ~1-2% to forward time
    if llm.is_moe:
        forward_time_s *= 1.015  # 1.5% routing overhead
    
    # Calculate activation memory (from ZeRO paper [3])
    # Activations must be stored for backward pass gradient computation
    # Formula: batch × seq × hidden × layers × bytes_per_element
    activation_memory_bytes: int = batch_size * seq_length * llm.d_model * llm.n_layers * 2  # 2 bytes for fp16
    activation_memory_gb: float = activation_memory_bytes / (2**30)
    
    return forward_time_s, activation_memory_gb


