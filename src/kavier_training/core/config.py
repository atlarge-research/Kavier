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
    
    # MFU improves with larger batches (better parallelism)
    # Batch-size scaling factors
    if total_work < 1024:
        # Very small batches: poor GPU utilization
        mfu = base_mfu * 0.6
    elif total_work < 4096:
        # Small batches: moderate utilization
        mfu = base_mfu * 0.8
    elif total_work < 16384:
        # Medium batches: good utilization
        mfu = base_mfu * 0.95
    else:
        # Large batches: near-optimal utilization
        mfu = base_mfu * 1.0
    
    return mfu


# Training overhead (kernel launch, synchronization)
TRAINING_OVERHEAD_S = 0.05  # 50ms per step

# Backward pass multiplier (from Megatron-LM paper)
BACKWARD_MULTIPLIER = 2.0

# Made with Bob
