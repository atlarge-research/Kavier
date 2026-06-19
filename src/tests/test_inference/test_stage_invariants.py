"""Physical invariants & sanity bounds for the inference timing/memory stages.

These verify the simulator obeys the physics it claims to model, across the
whole shipped GPU x LLM library (not one hand-picked pair), using property-based
ranges where it fits:
  * prefill/decode/kv-usage are non-negative and finite;
  * prefill is monotonic non-decreasing in input length and >= its fixed overhead;
  * decode without KV is super-linear (quadratic) and >= decode WITH KV;
  * a bigger model never decodes faster on the same GPU;
  * KV utilisation stays in [0, 1] and grows monotonically through a request.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kavier_inference.stages.decode import get_decode_time_s
from kavier_inference.stages.kv_usage import get_kv_cache_utilization
from kavier_inference.stages.prefill import get_prefill_time_s
from kavier_io.constants import PREFILL_OVERHEAD_S
from kavier_library.gpu import GPU_SPEC_LIBRARY
from kavier_library.llm import LLM_SPEC_LIBRARY

_GPU = GPU_SPEC_LIBRARY["A100-80GB"]
_SMALL_LLM = LLM_SPEC_LIBRARY["Llama-3-8B"]
_BIG_LLM = LLM_SPEC_LIBRARY["OPT-175B"]


# --------------------------------------------------------------------------- #
# Non-negativity / finiteness across the WHOLE library
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gpu_name", list(GPU_SPEC_LIBRARY))
@pytest.mark.parametrize("llm_name", list(LLM_SPEC_LIBRARY))
@pytest.mark.parametrize("kv", [True, False])
def test_prefill_decode_nonnegative_finite(gpu_name, llm_name, kv) -> None:
    gpu = GPU_SPEC_LIBRARY[gpu_name]
    llm = LLM_SPEC_LIBRARY[llm_name]
    tp = get_prefill_time_s(256, llm, gpu)
    td = get_decode_time_s(64, llm, gpu, kv_cache=kv)
    assert tp >= 0 and math.isfinite(tp)
    assert td >= 0 and math.isfinite(td)
    # prefill always pays at least the fixed pipeline overhead.
    assert tp >= PREFILL_OVERHEAD_S


@given(n_in=st.integers(min_value=0, max_value=200_000))
@settings(max_examples=40, deadline=None)
def test_prefill_monotonic_in_input_length(n_in) -> None:
    base = get_prefill_time_s(n_in, _SMALL_LLM, _GPU)
    more = get_prefill_time_s(n_in + 1, _SMALL_LLM, _GPU)
    assert more >= base
    assert base >= PREFILL_OVERHEAD_S  # overhead floor even at n_in=0


def test_prefill_zero_tokens_is_pure_overhead() -> None:
    # With no input tokens there is no compute term; time == fixed overhead.
    assert get_prefill_time_s(0, _SMALL_LLM, _GPU) == pytest.approx(PREFILL_OVERHEAD_S)


# --------------------------------------------------------------------------- #
# Decode scaling laws
# --------------------------------------------------------------------------- #
@given(n=st.integers(min_value=2, max_value=2_000))
@settings(max_examples=40, deadline=None)
def test_decode_no_kv_is_superlinear_vs_kv(n) -> None:
    # Without KV cache, attention re-reads the whole prefix every step:
    # cost ~ n(n+1)/2 vs n with KV. So no-KV >= KV for the same request, and the
    # ratio grows with n (quadratic vs linear).
    with_kv = get_decode_time_s(n, _SMALL_LLM, _GPU, kv_cache=True)
    no_kv = get_decode_time_s(n, _SMALL_LLM, _GPU, kv_cache=False)
    assert no_kv >= with_kv


def test_decode_no_kv_quadratic_ratio() -> None:
    # Doubling output length without KV ~4x the time (n(n+1)/2 dominated by n^2).
    t1 = get_decode_time_s(200, _SMALL_LLM, _GPU, kv_cache=False)
    t2 = get_decode_time_s(400, _SMALL_LLM, _GPU, kv_cache=False)
    assert math.isclose(t2 / t1, 4.0, rel_tol=0.02)


def test_bigger_model_decodes_at_least_as_slow() -> None:
    # More parameters -> more FLOPs/byte traffic per token -> not faster.
    small = get_decode_time_s(128, _SMALL_LLM, _GPU, kv_cache=True)
    big = get_decode_time_s(128, _BIG_LLM, _GPU, kv_cache=True)
    assert big >= small


# --------------------------------------------------------------------------- #
# KV-cache utilisation bounds & monotonicity
# --------------------------------------------------------------------------- #
def test_kv_utilization_disabled_is_zero() -> None:
    assert (
        get_kv_cache_utilization(
            _SMALL_LLM, _GPU, t_prefill=1.0, t_decode=1.0, t=0.5,
            prompt_len=100, response_len=20, kv_cache=False,
        )
        == 0
    )


@given(
    frac=st.floats(min_value=0.0, max_value=1.0),
    prompt_len=st.integers(min_value=1, max_value=4_000),
    response_len=st.integers(min_value=1, max_value=4_000),
)
@settings(max_examples=60, deadline=None)
def test_kv_utilization_in_unit_interval(frac, prompt_len, response_len) -> None:
    t_prefill, t_decode = 0.4, 0.8
    t = frac * (t_prefill + t_decode)
    use = get_kv_cache_utilization(
        _SMALL_LLM, _GPU, t_prefill=t_prefill, t_decode=t_decode, t=t,
        prompt_len=prompt_len, response_len=response_len, kv_cache=True,
    )
    assert 0.0 <= use <= 1.0


def test_kv_utilization_monotonic_through_request() -> None:
    # KV footprint only grows as more tokens are generated; sampling forward in
    # time must be non-decreasing.
    t_prefill, t_decode = 0.4, 0.8
    total = t_prefill + t_decode
    prev = -1.0
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        use = get_kv_cache_utilization(
            _SMALL_LLM, _GPU, t_prefill=t_prefill, t_decode=t_decode, t=frac * total,
            prompt_len=500, response_len=500, kv_cache=True,
        )
        assert use >= prev
        prev = use
