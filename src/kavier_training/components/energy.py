"""
Energy and power modeling for training workloads.

References:
- Desrochers et al. 2016: "A Validation of DRAM RAPL Power Models"
- NVIDIA Data Center GPU documentation
- Chowdhery et al. 2022: "PaLM" - GPU utilization analysis
"""

from library.specs.GPUSpec import GPUSpec


def calculate_gpu_power(
    compute_utilization: float,
    memory_utilization: float,
    gpu_spec: GPUSpec
) -> float:
    """
    Calculate GPU power consumption based on utilization.
    
    GPU power scales with both compute and memory utilization.
    Power model: P = P_idle + (P_max - P_idle) * max(compute_util, memory_util)
    
    Args:
        compute_utilization: Fraction of peak FLOPS utilized (0-1)
        memory_utilization: Fraction of peak bandwidth utilized (0-1)
        gpu_spec: GPU specifications
        
    Returns:
        Power consumption in watts
        
    References:
        - NVIDIA GPU power management documentation
        - Desrochers et al. 2016: DRAM power modeling
    """
    # Idle power is ~20-30% of TDP for modern GPUs
    idle_power = gpu_spec.base_power_w * 0.25
    max_power = gpu_spec.base_power_w
    
    # Power scales with max of compute or memory utilization
    # (whichever is the bottleneck)
    active_utilization = max(compute_utilization, memory_utilization)
    
    # Linear power scaling model
    power_watts = idle_power + (max_power - idle_power) * active_utilization
    
    return power_watts


def calculate_compute_utilization(
    achieved_flops: float,
    peak_flops: float
) -> float:
    """
    Calculate GPU compute utilization.
    
    Args:
        achieved_flops: Achieved FLOPS
        peak_flops: Peak FLOPS capability
        
    Returns:
        Utilization fraction (0-1)
    """
    return min(1.0, achieved_flops / peak_flops) if peak_flops > 0 else 0.0


def calculate_memory_utilization(
    memory_bandwidth_used: float,
    peak_bandwidth: float
) -> float:
    """
    Calculate GPU memory bandwidth utilization.
    
    Args:
        memory_bandwidth_used: Bandwidth used (GB/s)
        peak_bandwidth: Peak bandwidth (GB/s)
        
    Returns:
        Utilization fraction (0-1)
    """
    return min(1.0, memory_bandwidth_used / peak_bandwidth) if peak_bandwidth > 0 else 0.0


def estimate_memory_bandwidth_usage(
    model_params: float,
    batch_size: int,
    seq_length: int,
    step_time_s: float
) -> float:
    """
    Estimate memory bandwidth usage during training.
    
    Training requires loading:
    - Model parameters (forward + backward)
    - Gradients
    - Optimizer states (momentum, variance for Adam)
    - Activations
    
    Args:
        model_params: Number of model parameters
        batch_size: Training batch size
        seq_length: Sequence length
        step_time_s: Time for one training step
        
    Returns:
        Estimated bandwidth usage in GB/s
    """
    # Bytes per parameter (fp16 = 2 bytes)
    bytes_per_param = 2
    
    # Memory traffic per step:
    # - Parameters: 2x (forward + backward)
    # - Gradients: 1x
    # - Optimizer states: 2x (Adam has momentum + variance)
    # - Activations: ~batch_size * seq_length * hidden_dim (simplified)
    param_traffic = model_params * bytes_per_param * (2 + 1 + 2)
    
    # Activation traffic (rough estimate)
    activation_traffic = batch_size * seq_length * 4096 * bytes_per_param
    
    total_bytes = param_traffic + activation_traffic
    total_gb = total_bytes / (1024**3)
    
    # Bandwidth = data / time
    bandwidth_gbs = total_gb / step_time_s if step_time_s > 0 else 0.0
    
    return bandwidth_gbs

# Made with Bob
