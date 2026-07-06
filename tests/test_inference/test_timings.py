"""Independent-oracle tests for the inference roofline timing physics.

Source under test:
  - kavier.sdk.inference.stages.prefill.get_prefill_time_s
  - kavier.sdk.inference.stages.decode.get_decode_time_s

Fixed physics constants (kavier.sdk.io.constants), written here as raw numbers so
the oracles do NOT re-import the same symbols the code uses:
    COMPUTE_EFFICIENCY = 0.30 ,  MEMORY_EFFICIENCY = 0.60 ,  PREFILL_OVERHEAD_S = 0.025
"""

from hypothesis import given
from hypothesis import strategies as st
from pytest import approx

from kavier.sdk.inference.stages.decode import get_decode_time_s
from kavier.sdk.inference.stages.prefill import get_prefill_time_s
from kavier.sdk.library.specs.GPUSpec import GPUSpec
from kavier.sdk.library.specs.LLMSpec import LLMSpec

# --- Reference hardware/model -------------------------------------------------
# Memory-bound decode GPU: 125 TFLOP/s tensor cores, 600 GB/s bandwidth.
gpu = GPUSpec(
    gpu_name="A10",
    memory_bandwidth_gbps=600,
    fp_16_tensor_core_tflops=125,
    gpu_cores=9216,
    gpu_core_max_mhz=1695,
    memory_gb=24,
    base_power_w=150,
)
# Same GPU but starved of compute (1 TFLOP/s) so the decode roofline flips to
# the COMPUTE-bound branch of the max() (see test_decode_roofline_takes_max).
gpu_compute_starved = GPUSpec(
    gpu_name="starved",
    memory_bandwidth_gbps=600,
    fp_16_tensor_core_tflops=1,
    gpu_cores=1,
    gpu_core_max_mhz=1,
    memory_gb=1,
    base_power_w=1,
)

# Dense 7B model, 2 bytes/param (fp16). active_params defaults to m_params.
llm = LLMSpec(
    llm_name="Test-7B",
    n_layers=32,
    n_heads=32,
    d_head=128,
    d_model=4096,
    p_bytes=2,
    m_params=7e9,
)
# Same weights but MoE: only 1e9 of the 7e9 params are active per token.
llm_moe = LLMSpec(
    llm_name="Test-MoE",
    n_layers=32,
    n_heads=32,
    d_head=128,
    d_model=4096,
    p_bytes=2,
    m_params=7e9,
    active_params=1e9,
)


# ============================ PREFILL (compute-bound) =========================


def test_prefill_at_zero_tokens_is_fixed_overhead():
    # With no tokens the compute term vanishes, leaving only the fixed launch
    # overhead PREFILL_OVERHEAD_S = 0.025 s. Falsifies if the overhead term is
    # dropped (-> 0.0) or the FLOP term is not proportional to n_in.
    assert get_prefill_time_s(0, llm, gpu) == approx(0.025)


def test_prefill_time_matches_hand_derived_roofline():
    # time = 0.025 + n_in * (2*active_params) / (TFLOPS*1e12*0.30)
    # per-token compute = 2*7e9 / (125e12*0.30) = 1.4e10 / 3.75e13 = 3.73333e-4 s
    #   n=1000 -> 0.025 + 0.373333 = 0.398333 s
    #   n=2000 -> 0.025 + 0.746667 = 0.771667 s
    # Pins both intercept (0.025) and slope; falsifies on any efficiency (0.30)
    # or FLOP-factor (2x) perturbation.
    assert get_prefill_time_s(1_000, llm, gpu) == approx(0.3983333333, abs=1e-9)
    assert get_prefill_time_s(2_000, llm, gpu) == approx(0.7716666667, abs=1e-9)


def test_prefill_uses_active_params_not_total():
    # The MoE model has active_params = 1e9 vs the dense model's 7e9, same total
    # weights. Above the shared fixed overhead, prefill compute scales with the
    # params a token actually touches, so the ratio must be 1e9/7e9 = 1/7.
    # Falsifies if the code used m_params (ratio would be 1.0).
    dense_compute = get_prefill_time_s(1_000, llm, gpu) - 0.025
    moe_compute = get_prefill_time_s(1_000, llm_moe, gpu) - 0.025
    assert moe_compute / dense_compute == approx(1.0 / 7.0)


# ============================ DECODE (memory-bound) ==========================


def test_decode_with_kv_is_linear_and_memory_bound():
    # Decode is bandwidth-bound: per-token = p_bytes*active_params/(BW*0.60)
    #   = 2*7e9 / (600e9*0.60) = 1.4e10 / 3.6e11 = 7/180 = 0.0388889 s/token.
    # With KV cache the cost is linear: 90 tokens * 7/180 = 3.5 s exactly.
    # Falsifies if MEMORY_EFFICIENCY changes, or if the (smaller) compute-bound
    # term were used instead of the memory-bound one.
    assert get_decode_time_s(90, llm, gpu, kv_cache=True) == approx(3.5, abs=1e-9)


def test_decode_roofline_takes_max_of_compute_and_memory():
    # On the compute-starved GPU (1 TFLOP/s) the compute-bound per-token time
    #   2*7e9 / (1e12*0.30) = 1.4e10/3e11 = 7/150 = 0.0466667 s
    # exceeds the memory-bound one (7/180 = 0.0388889 s), so the roofline max()
    # must select compute. decode(1, kv) == 7/150. A min() (or memory-only)
    # implementation would give 7/180 instead.
    t = get_decode_time_s(1, llm, gpu_compute_starved, kv_cache=True)
    assert t == approx(7.0 / 150.0, abs=1e-9)
    assert t > 7.0 / 180.0  # strictly above the memory-bound value


def test_decode_uses_active_params_not_total():
    # Memory-bound per-token cost scales linearly with active_params, so the MoE
    # (1e9) vs dense (7e9) decode ratio must be 1/7 for identical token counts.
    # Falsifies if either the FLOP or the memory term used m_params.
    ratio = get_decode_time_s(50, llm_moe, gpu, kv_cache=True) / get_decode_time_s(50, llm, gpu, kv_cache=True)
    assert ratio == approx(1.0 / 7.0)


@given(n=st.integers(min_value=1, max_value=5_000))
def test_decode_without_kv_is_triangular_relative_to_kv(n):
    # Without a KV cache every step re-reads the whole growing context, giving a
    # triangular n(n+1)/2 cost vs the linear n cost with KV. Their ratio is an
    # exact identity independent of the per-token time: (n(n+1)/2)/n = (n+1)/2.
    # At n=1 both branches coincide (1*2/2 == 1). Falsifies if the no-KV branch
    # is linear (ratio -> 1) or otherwise non-triangular.
    with_kv = get_decode_time_s(n, llm, gpu, kv_cache=True)
    without_kv = get_decode_time_s(n, llm, gpu, kv_cache=False)
    assert without_kv / with_kv == approx((n + 1) / 2.0, rel=1e-9)
