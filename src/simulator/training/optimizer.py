"""
Optimizer step simulation for training (AdamW).

References:
- Kingma & Ba 2015: "Adam: A Method for Stochastic Optimization" - Original Adam algorithm
- Loshchilov & Hutter 2019: "Decoupled Weight Decay Regularization" - AdamW variant
- Rajbhandari et al. 2020: "ZeRO" - Optimizer state memory analysis
"""

from simulator.performance.util.specs import GPUSpec, LLMSpec


def calculate_optimizer_step(
    llm: LLMSpec,
    gpu: GPUSpec,
) -> tuple[float, float]:
    """
    Calculate optimizer step time and memory for AdamW (full fine-tuning).
    
    AdamW maintains two state tensors per parameter:
    - Momentum (first moment): exponential moving average of gradients
    - Variance (second moment): exponential moving average of squared gradients
    
    The optimizer step is memory-bandwidth limited, not compute-limited.
    It must read gradients, read/update states, and write new parameters.
    
    Time Calculation:
    - Memory-bandwidth limited operation
    - Must transfer: gradients (read) + 2 states (read/write) + parameters (write)
    - Total: 4 reads + 2 writes = 6 memory operations per parameter
    - Formula: T = (6 × params × bytes_per_param) / memory_bandwidth
    
    Memory Calculation:
    - Based on Rajbhandari et al. 2020 (ZeRO paper) [3]
    - Optimizer states stored in fp32 for numerical stability
    - Formula: optimizer_memory = 2 × params × 4 bytes (fp32)
    
    Args:
        llm: LLM specifications
        gpu: GPU specifications
        
    Returns:
        Tuple of (optimizer_time_s, optimizer_memory_gb)
        
    References:
        [1] Kingma & Ba 2015: "Adam: A Method for Stochastic Optimization"
        [2] Loshchilov & Hutter 2019: "Decoupled Weight Decay Regularization" (AdamW)
        [3] Rajbhandari et al. 2020: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
    """
    # Optimizer memory: 2 state tensors (momentum + variance) in fp32 (from ZeRO [3])
    optimizer_memory_bytes = 2 * llm.m_params * 4  # 4 bytes for fp32
    optimizer_memory_gb = optimizer_memory_bytes / (1024**3)
    
    # Optimizer time: memory-bandwidth limited
    # Memory operations per parameter:
    # - Read gradient (fp16): 2 bytes
    # - Read momentum state (fp32): 4 bytes
    # - Read variance state (fp32): 4 bytes
    # - Write updated momentum (fp32): 4 bytes
    # - Write updated variance (fp32): 4 bytes
    # - Write updated parameter (fp16): 2 bytes
    # Total: 20 bytes per parameter
    bytes_per_param_transfer = 20
    total_bytes_transferred = llm.m_params * bytes_per_param_transfer
    
    # Time = bytes / bandwidth
    optimizer_time_s = total_bytes_transferred / gpu.bandwidth_bps
    
    return optimizer_time_s, optimizer_memory_gb

# Made with Bob
