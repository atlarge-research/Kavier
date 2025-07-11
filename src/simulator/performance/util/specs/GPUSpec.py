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
    ):
        """
        Initialize GPU specifications for LLM inference simulation.

        Parameters:
            gpu_name (str): Name of the GPU.
            memory_bandwidth_gbps (int): Memory bandwidth in GB/s.
            fp_16_tensor_core_tflops (int): FP16 tensor core performance in TFLOPS.
            gpu_cores (int): Number of GPU cores.
            boost_clock_mhz (int): Boost clock speed in MHz.
            prefill_speed_tps (float): Prefill speed in tokens per second.
            decode_speed_tps (float): Decode speed in tokens per second.
            num_streaming_multiprocessors (int): Number of streaming multiprocessors.
            flops_per_token_1b (float): FLOPs required per token.
            memory_gb (float): GPU memory in GB.
            base_power_w (float): Base power consumption in watts.

            -- some were removed as they are not used in the simulation, may be added later --
        """
        self.name = gpu_name
        self.cores = gpu_cores
        self.fp_16_tensor_core_tflops = fp_16_tensor_core_tflops
        self.bandwidth_bps = memory_bandwidth_gbps * 1e9  # Convert to bytes/sec
        self.memory_gb = memory_gb
        self.core_max_mhz = gpu_core_max_mhz
