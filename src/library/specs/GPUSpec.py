class GPUSpec:
    def __init__(
            self,
            gpu_name: str,
            memory_bandwidth_gbps: int,
            fp_16_tensor_core_tflops: int,
            gpu_cores: int,
            memory_gb: float,
            gpu_core_max_mhz,
            base_power_w: float,
            mfu_factor: float = 0.45,
    ):
        """
        Initialize GPU specifications for LLM inference and training simulation.

        Parameters:
            gpu_name (str): Name of the GPU.
            memory_bandwidth_gbps (int): Memory bandwidth in GB/s.
            fp_16_tensor_core_tflops (int): FP16 tensor core performance in TFLOPS.
            gpu_cores (int): Number of GPU cores.
            memory_gb (float): GPU memory in GB.
            gpu_core_max_mhz (int): GPU core max clock speed in MHz.
            base_power_w (float): Base power consumption in watts.
            mfu_factor (float): Model FLOPs Utilization factor (fraction of peak FLOPS achieved).
                Default 0.45. Represents achieved_FLOPS / peak_FLOPS during training.
                Varies by architecture due to memory bandwidth and kernel efficiency:
                - Ampere (A100): ~0.40-0.45 (memory-bound)
                - Hopper (H100): ~0.15-0.20 (higher peak FLOPS, still memory-bound)
                - Ada/Lovelace: ~0.38-0.48
                
                Reference: Chowdhery et al. 2022 (PaLM paper) - "Model FLOPs Utilization"

            -- some parameters were removed as they are not used in the simulation, may be added later --
        """
        self.name = gpu_name
        self.cores = gpu_cores
        self.fp_16_tensor_core_tflops = fp_16_tensor_core_tflops
        self.bandwidth_bps = memory_bandwidth_gbps * 1e9  # Convert to bytes/sec
        self.memory_gb = memory_gb
        self.core_max_mhz = gpu_core_max_mhz
        self.mfu_factor = mfu_factor
