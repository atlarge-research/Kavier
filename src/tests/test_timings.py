import math
import random

from simulator.performance.util.decode import get_decode_time_s
from simulator.performance.util.prefill import get_prefill_time_s
from simulator.performance.util.specs import GPUSpec, LLMSpec

gpu = GPUSpec(
    gpu_name="A10",
    memory_bandwidth_gbps=600,
    fp_16_tensor_core_tflops=125,
    gpu_cores=9216,
    gpu_core_max_mhz=1695,
    memory_gb=24,
    base_power_w=150,
)
llm = LLMSpec(
    llm_name="Test-7B", n_layers=32, n_heads=32, d_head=128, d_model=4096, p_bytes=2, m_params=7e9
)


def test_prefill_scales_linearly():
    t1 = get_prefill_time_s(1_000, llm, gpu)
    t2 = get_prefill_time_s(2_000, llm, gpu)
    assert t1 < t2
    assert math.isclose(t1, 0.4, rel_tol=0.05)
    assert math.isclose(t2, 0.77, rel_tol=0.05)


def test_decode_scales_quadratically_without_kv():
    n = random.randint(50, 100)
    t1 = get_decode_time_s(n, llm, gpu, kv_cache=False)
    t2 = get_decode_time_s(2 * n, llm, gpu, kv_cache=False)
    assert math.isclose(t2 / t1, 4.0, rel_tol=0.05)


def test_decode_scales_linearly_with_kv():
    n = random.randint(50, 100)
    t1 = get_decode_time_s(n, llm, gpu, kv_cache=True)
    t2 = get_decode_time_s(2 * n, llm, gpu, kv_cache=True)
    assert math.isclose(t2 / t1, 2.0, rel_tol=0.02)
