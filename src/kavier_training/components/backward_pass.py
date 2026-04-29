"""
Backward pass simulation for training.

References:
- Shoeybi et al. 2019: "Megatron-LM" - Backward pass ~2x forward pass FLOPs
- Narayanan et al. 2021: "Pipeline Parallelism" - Uses 2x multiplier in timing models
- Rajbhandari et al. 2020: "ZeRO" - Gradient memory equals parameter memory
"""

from library.specs.LLMSpec import LLMSpec


def calculate_backward_pass(
    forward_time_s: float,
    llm: LLMSpec,
) -> tuple[float, float]:
    """
    Calculate backward pass time and gradient memory for training.
    
    The backward pass computes gradients for all model parameters using
    backpropagation. Standard practice in literature is to model this as
    approximately 2x the forward pass computation.
    
    Time Calculation:
    - Based on Shoeybi et al. 2019 (Megatron-LM) [1] and Narayanan et al. 2021 [2]
    - Backward pass performs ~2x the FLOPs of forward pass
    - Formula: T_backward = 2.0 × T_forward
    
    Memory Calculation:
    - Based on Rajbhandari et al. 2020 (ZeRO paper) [3]
    - Gradient memory equals parameter memory
    - Formula: gradient_memory = model_params × bytes_per_param
    
    Args:
        forward_time_s: Forward pass time in seconds
        llm: LLM specifications
        
    Returns:
        Tuple of (backward_time_s, gradient_memory_gb)
        
    References:
        [1] Shoeybi et al. 2019: "Megatron-LM: Training Multi-Billion Parameter Language Models"
        [2] Narayanan et al. 2021: "Efficient Large-Scale Language Model Training on GPU Clusters"
        [3] Rajbhandari et al. 2020: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
    """
    # Backward pass time: 2x forward pass (from Megatron-LM [1] and Pipeline Parallelism [2])
    backward_time_s = 2.0 * forward_time_s
    
    # Gradient memory: same size as model parameters (from ZeRO paper [3])
    # Each parameter needs a gradient of the same size
    gradient_memory_bytes = llm.m_params * llm.p_bytes
    gradient_memory_gb = gradient_memory_bytes / (1024**3)
    
    return backward_time_s, gradient_memory_gb


