"""
Training-specific configuration constants.

Based on empirical observations and roofline model analysis.
"""

from library.specs.GPUSpec import GPUSpec


def get_training_compute_efficiency(
    batch_size: int,
    seq_length: int,
    gpu_spec: GPUSpec
) -> float:
    """
    Compute GPU MFU (Model FLOPs Utilization) for training.
    
    MFU = achieved_FLOPS / peak_FLOPS
    
    Training achieves lower MFU than inference due to:
    - Activation checkpointing (memory-compute tradeoff)
    - Gradient accumulation overhead
    - Optimizer state updates
    
    MFU increases with batch size due to better GPU utilization.
    Based on roofline model (Williams et al. 2009).
    
    Args:
        batch_size: Training batch size
        seq_length: Sequence length in tokens
        gpu_spec: GPU specification with architecture-specific MFU factor
        
    Returns:
        MFU factor (0-1) representing fraction of peak FLOPS achieved
        
    References:
        - Williams et al. 2009: "Roofline: An Insightful Visual Performance Model"
        - Chowdhery et al. 2022: "PaLM: Scaling Language Modeling with Pathways" (MFU definition)
    """
    # Total work: batch_size × seq_length
    total_work = batch_size * seq_length
    
    # Use GPU-specific MFU factor (varies by architecture)
    # Calibrated via least-squares optimization on validation data
    base_mfu = gpu_spec.mfu_factor
    
    # Batch-size scaling: MFU improves with larger batches (better parallelism)
    if batch_size == 1:
        batch_scale = 0.5  # Single sample: very poor utilization
    elif batch_size == 2:
        batch_scale = 0.65
    elif batch_size == 4:
        batch_scale = 0.8
    elif batch_size <= 8:
        batch_scale = 0.9
    elif batch_size <= 16:
        batch_scale = 0.95
    else:
        batch_scale = 1.0  # Large batches: optimal
    
    # Sequence-length scaling: Short sequences have higher overhead
    if seq_length <= 512:
        seq_scale = 0.7  # High kernel launch overhead relative to compute
    elif seq_length <= 1024:
        seq_scale = 0.85
    elif seq_length <= 2048:
        seq_scale = 0.95
    else:
        seq_scale = 1.0  # Long sequences: overhead amortized
    
    # Combined MFU with both scaling factors
    mfu = base_mfu * batch_scale * seq_scale
    
    return mfu


# Training overhead (kernel launch, synchronization)
TRAINING_OVERHEAD_S = 0.05  # 50ms per step

# Backward pass multiplier (from Megatron-LM paper)
BACKWARD_MULTIPLIER = 2.0

# Made with Bob
