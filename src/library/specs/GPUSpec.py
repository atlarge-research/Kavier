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
        network_bandwidth_gbps: float = 400.0,
    ):
        self.name = gpu_name
        self.cores = gpu_cores
        self.fp_16_tensor_core_tflops = fp_16_tensor_core_tflops
        self.bandwidth_bps = memory_bandwidth_gbps * 1e9
        self.memory_gb = memory_gb
        self.core_max_mhz = gpu_core_max_mhz
        self.base_power_w = base_power_w
        self.mfu_factor = mfu_factor
        self.network_bandwidth_gbps = network_bandwidth_gbps
