"""Physics oracles for the inference timing/memory stages (prefill, decode roofline, KV utilisation).

Every expected value here is either hand-derived from first principles with a literal number
(so a perturbed constant/exponent goes red), an exact functional identity, or a catalog-wide
scaling law — never a snapshot of the code's own output.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kavier.sdk.inference.stages.decode import get_decode_time_s
from kavier.sdk.inference.stages.prefill import get_prefill_time_s
from kavier.sdk.io.constants import PREFILL_OVERHEAD_S
from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY
from kavier.sdk.library.llm import LLM_SPEC_LIBRARY
from kavier.sdk.library.specs.GPUSpec import GPUSpec
from kavier.sdk.library.specs.LLMSpec import LLMSpec

_GPU = GPU_SPEC_LIBRARY["A100-80GB"]  # 312 fp16 TFLOPS, 1935 GB/s, 80 GiB
_SMALL_LLM = LLM_SPEC_LIBRARY["Llama-3-8B"]  # 8e9 active params, 32 layers, d_model 4096, fp16
_BIG_LLM = LLM_SPEC_LIBRARY["OPT-175B"]  # 175e9 active params, fp16


# --------------------------------------------------------------------------- prefill


def test_prefill_zero_tokens_is_pure_overhead() -> None:
    # n_in=0 kills the compute term, leaving only the fixed launch overhead.
    assert get_prefill_time_s(0, _SMALL_LLM, _GPU) == pytest.approx(PREFILL_OVERHEAD_S)


def test_prefill_matches_hand_derived_flops() -> None:
    # 2 FLOPs/param/token forward pass; A100 delivers 312e12 * 0.30 effective FLOP/s.
    #   overhead + 256 * (2 * 8e9) / (312e12 * 0.30)
    #   = 0.025 + 4.096e12 / 9.36e13 = 0.025 + 0.043760683... = 0.068760683...
    assert get_prefill_time_s(256, _SMALL_LLM, _GPU) == pytest.approx(0.06876068376068376)


@given(
    a=st.integers(min_value=0, max_value=100_000),
    b=st.integers(min_value=0, max_value=100_000),
)
@settings(max_examples=50, deadline=None)
def test_prefill_compute_term_is_additive_in_tokens(a, b) -> None:
    # prefill(n) = overhead + k*n, so prefill(a+b) == prefill(a) + prefill(b) - overhead.
    # Fails for any nonlinearity (e.g. an n^2 or sqrt(n) compute term).
    lhs = get_prefill_time_s(a + b, _SMALL_LLM, _GPU)
    rhs = get_prefill_time_s(a, _SMALL_LLM, _GPU) + get_prefill_time_s(b, _SMALL_LLM, _GPU)
    assert lhs == pytest.approx(rhs - PREFILL_OVERHEAD_S)


# --------------------------------------------------------------------------- decode


def test_decode_kv_matches_hand_derived_memory_bound() -> None:
    # For fp16 on the A100 the roofline is memory-bound (bandwidth term ~80x the compute term):
    #   per-token = p_bytes * active / (bw * 0.60) = 2 * 8e9 / (1935e9 * 0.60)
    #             = 1.6e10 / 1.161e12 = 0.0137812230835...
    #   128 tokens with KV = 128 * that = 1.7639965546942...
    assert get_decode_time_s(128, _SMALL_LLM, _GPU, kv_cache=True) == pytest.approx(1.7639965546942291)


@given(n=st.integers(min_value=1, max_value=5_000))
@settings(max_examples=40, deadline=None)
def test_decode_kv_is_linear_in_output_length(n) -> None:
    # With KV cache each step is independent, so doubling n_out exactly doubles the time.
    one = get_decode_time_s(n, _SMALL_LLM, _GPU, kv_cache=True)
    two = get_decode_time_s(2 * n, _SMALL_LLM, _GPU, kv_cache=True)
    assert two == pytest.approx(2.0 * one)


@pytest.mark.parametrize("gpu_name", list(GPU_SPEC_LIBRARY))
@pytest.mark.parametrize("llm_name", list(LLM_SPEC_LIBRARY))
def test_decode_no_kv_is_triangular_multiple_of_kv(gpu_name, llm_name) -> None:
    # No-KV re-reads the whole prefix every step: no_kv = n(n+1)/2 * tpt vs kv = n * tpt,
    # so no_kv/kv == (n+1)/2 exactly, independent of GPU/LLM. n=201 -> 101.
    # A plain n^2 (instead of n(n+1)/2) would give ratio n=201, not 101.
    gpu = GPU_SPEC_LIBRARY[gpu_name]
    llm = LLM_SPEC_LIBRARY[llm_name]
    kv = get_decode_time_s(201, llm, gpu, kv_cache=True)
    no_kv = get_decode_time_s(201, llm, gpu, kv_cache=False)
    assert no_kv == pytest.approx(kv * 101.0)


def test_decode_scales_linearly_with_active_params() -> None:
    # Both models are memory-bound on the A100, and the memory term is linear in active_params,
    # so the ratio is exactly 175e9 / 8e9 = 21.875 (the per-token constants cancel).
    small = get_decode_time_s(128, _SMALL_LLM, _GPU, kv_cache=True)
    big = get_decode_time_s(128, _BIG_LLM, _GPU, kv_cache=True)
    assert big == pytest.approx(small * 21.875)


def test_decode_roofline_takes_compute_bound_when_it_dominates() -> None:
    # All shipped GPUs are memory-bound for fp16, so build a synthetic GPU with tiny compute
    # (1 TFLOPS) and enormous bandwidth (1e6 GB/s) to force the compute arm of max() to win.
    #   compute = 2 * 1e9 / (1e12 * 0.30) = 6.666...e-3  (>> memory 2 * 1e9 / (1e15 * 0.60))
    #   10 tokens with KV = 10 * 6.666...e-3 = 0.06666...
    fast_bw_gpu = GPUSpec(
        gpu_name="synthetic-compute-bound",
        memory_bandwidth_gbps=1_000_000,  # 1e15 bytes/s
        fp_16_tensor_core_tflops=1,
        gpu_cores=1,
        memory_gb=80,
        gpu_core_max_mhz=1000,
        base_power_w=400,
    )
    llm = LLMSpec("synthetic-1b", n_layers=1, d_model=1, p_bytes=2, m_params=1e9, n_heads=1, d_head=1)
    got = get_decode_time_s(10, llm, fast_bw_gpu, kv_cache=True)
    assert got == pytest.approx(0.06666666666666667)
    # And it must be the compute value, not the (tiny) memory value 10 * 3.333e-6 = 3.33e-5.
    assert got > 1e-3
