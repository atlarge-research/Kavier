"""
Training-specific configuration constants.

Based on empirical observations and roofline model analysis.
"""

from library.specs.GPUSpec import GPUSpec


# ============================================================================
# MFU Scaling Constants
# ============================================================================
# Batch scaling: MFU increases with batch_size because larger batches
# improve arithmetic intensity (compute / memory-access ratio).
# Reaches 1.0 around bs=64; small batches (bs<8) have significantly lower MFU.
BATCH_ALPHA = 0.0341
BATCH_BETA = 0.8147

# Sequence-length scaling: longer sequences amortise attention overhead.
SEQ_GAMMA = 0.1781
SEQ_DELTA = 3.5714

# ============================================================================
# Training Overhead Constants
# ============================================================================
# Training overhead (kernel launch, synchronization, logging)
# Based on PyTorch profiling studies and production workloads
TRAINING_OVERHEAD_S = 0.05  # 50ms per step

# Backward pass multiplier
# Reference: Shoeybi et al. 2019 "Megatron-LM: Training Multi-Billion Parameter Language Models"
# Backward pass requires ~2x forward pass compute (gradient computation)
BACKWARD_MULTIPLIER = 2.0


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
    import math
    
    # Use GPU-specific MFU factor (varies by architecture)
    # Calibrated via least-squares optimization on validation data
    from kavier_training.core.calibration import get_mfu_multiplier

    base_mfu = gpu_spec.mfu_factor * get_mfu_multiplier(gpu_spec.name)
    
    batch_scale = min(1.0, BATCH_ALPHA * math.log2(max(1, batch_size)) + BATCH_BETA)
    seq_scale = min(1.0, SEQ_GAMMA * math.log2(max(1, seq_length / 512)) + SEQ_DELTA)
    
    # Combined MFU with both scaling factors
    mfu = base_mfu * batch_scale * seq_scale
    
    return mfu

