"""
Training overhead calculation.

Accounts for data loading, logging, and other non-compute operations.
"""


def calculate_training_overhead(batch_size: int) -> float:
    """
    Calculate training overhead time per step.
    
    Training overhead includes:
    - Data loading and preprocessing
    - Logging and metrics collection
    - Synchronization between operations
    
    This is a simplified model that scales with batch size.
    
    Args:
        batch_size: Training batch size
        
    Returns:
        Overhead time in seconds
    """
    # Base overhead + per-sample overhead
    # These are empirical values that can be tuned
    base_overhead_s = 0.001  # 1ms base
    per_sample_overhead_s = 0.0001  # 0.1ms per sample
    
    return base_overhead_s + (batch_size * per_sample_overhead_s)

# Made with Bob
