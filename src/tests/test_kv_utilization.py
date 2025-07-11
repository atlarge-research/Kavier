from simulator.performance.util.kv_utilization import get_kv_cache_utilization
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
    llm_name="Tiny", n_layers=2, n_heads=2, d_head=64, d_model=128, p_bytes=2, m_params=1e6
)


def test_kv_usage_bounds():
    use = get_kv_cache_utilization(
        llm,
        gpu,
        t_prefill=1,
        t_decode=1,
        t=0.5,
        prompt_len=100,
        response_len=20,
        kv_cache=True,
    )
    assert 0 < use < 1
