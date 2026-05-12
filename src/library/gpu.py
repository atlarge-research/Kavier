from library.specs.GPUSpec import GPUSpec

# ============================================================================
# Kavier GPU Specifications Library
# ============================================================================
#
# This module contains detailed GPU specifications for physics-based simulation.
# These specs are calibrated for Kavier's analytical models and include parameters
# like MFU factors, network bandwidth, and memory bandwidth.
#
# NOTE: For simplified GPU specs used by power predictor and webapp, see:
#       common/hardware_specs.py
#
# The duplication is intentional - Kavier is a standalone physics simulator with
# its own calibrated parameters optimized for accuracy in LLM training simulation.
# ============================================================================

# ============================================================================
# INFERENCE GPUs (8 GPUs for Kavier Inference Baseline)
# ============================================================================

GPU_SPEC_LIBRARY = {
    "A10": GPUSpec(
        gpu_name="A10",
        memory_gb=24,
        memory_bandwidth_gbps=600,
        fp_16_tensor_core_tflops=125,
        gpu_cores=9216,
        gpu_core_max_mhz=1695,
        base_power_w=150,
        mfu_factor=0.40,# look more into MFU, potentially make it dynamic
    ),
    "A100-40GB": GPUSpec(
        gpu_name="A100-40GB",
        memory_gb=40,
        memory_bandwidth_gbps=1555,
        fp_16_tensor_core_tflops=312,
        gpu_cores=6912,
        gpu_core_max_mhz=1410,
        base_power_w=250,
        mfu_factor=0.42,
    ),
    "A100-80GB": GPUSpec(
        gpu_name="A100-80GB",
        memory_gb=80,
        memory_bandwidth_gbps=1935,
        fp_16_tensor_core_tflops=312,
        gpu_cores=6912,
        gpu_core_max_mhz=1410,
        base_power_w=300,
        mfu_factor=0.45,
    ),
    "L4": GPUSpec(
        gpu_name="L4",
        memory_gb=24,
        memory_bandwidth_gbps=300,
        fp_16_tensor_core_tflops=242,
        gpu_cores=7424,
        gpu_core_max_mhz=2040,
        base_power_w=72,
        mfu_factor=0.38,
    ),
    "L40S": GPUSpec(
        gpu_name="L40S",
        memory_gb=48,
        memory_bandwidth_gbps=864,
        fp_16_tensor_core_tflops=362,
        gpu_cores=18176,
        gpu_core_max_mhz=2520,
        base_power_w=350,
        mfu_factor=0.1042,  # Recalibrated: 0.48 / 4.599 = 0.1042
    ),
    "H100-PCIe": GPUSpec(
        gpu_name="H100-PCIe",
        memory_gb=80,
        memory_bandwidth_gbps=2000,
        fp_16_tensor_core_tflops=1513,
        gpu_cores=14592,
        gpu_core_max_mhz=1755,
        base_power_w=350,
        mfu_factor=0.1554,  # Calibrated via least-squares
    ),
    "H100-SXM": GPUSpec(
        gpu_name="H100-SXM",
        memory_gb=80,
        memory_bandwidth_gbps=3350,
        fp_16_tensor_core_tflops=1979,
        gpu_cores=16896,
        gpu_core_max_mhz=1830,
        base_power_w=700,
        mfu_factor=0.55,
    ),
    "H200 SXM": GPUSpec(
        gpu_name="H200 SXM",
        memory_gb=141,
        memory_bandwidth_gbps=4800,
        fp_16_tensor_core_tflops=1979,
        gpu_cores=16896,
        gpu_core_max_mhz=1785,
        base_power_w=700,
        mfu_factor=0.58,
    ),
    
    # ============================================================================
    # TRAINING GPUs (3 GPUs from ado-sfttrainer dataset for Training)
    # ============================================================================
    
    "NVIDIA-A100-80GB-PCIe": GPUSpec(
        gpu_name="NVIDIA-A100-80GB-PCIe",
        memory_gb=80,
        memory_bandwidth_gbps=1935,
        fp_16_tensor_core_tflops=312,
        gpu_cores=6912,
        gpu_core_max_mhz=1410,
        base_power_w=300,
        mfu_factor=0.4513,  # Calibrated via least-squares
        network_bandwidth_gbps=512.0,  # PCIe 4.0 x16: 64 GB/s = 512 Gbps
    ),
    "NVIDIA-H100-PCIe": GPUSpec(
        gpu_name="NVIDIA-H100-PCIe",
        memory_gb=80,
        memory_bandwidth_gbps=2000,
        fp_16_tensor_core_tflops=1513,
        gpu_cores=14592,
        gpu_core_max_mhz=1755,
        base_power_w=350,
        mfu_factor=0.1554,  # Calibrated via least-squares
        network_bandwidth_gbps=1024.0,  # PCIe 5.0 x16: 128 GB/s = 1024 Gbps
    ),
    "NVIDIA-A100-SXM4-80GB": GPUSpec(
        gpu_name="NVIDIA-A100-SXM4-80GB",
        memory_gb=80,
        memory_bandwidth_gbps=2039,
        fp_16_tensor_core_tflops=312,
        gpu_cores=6912,
        gpu_core_max_mhz=1410,
        base_power_w=400,
        mfu_factor=0.4513,  # Reset to PCIe baseline for recalibration
        network_bandwidth_gbps=4800.0,  # NVLink 3.0: 600 GB/s = 4800 Gbps
    ),
}
