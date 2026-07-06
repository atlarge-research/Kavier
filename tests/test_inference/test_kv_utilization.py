"""Tests for the analytical KV-cache memory-utilization model (stages/kv_usage.py).

Independent-oracle setup
------------------------
The KV cache stores a Key and a Value tensor (factor 2) for every layer, each
``d_model`` wide, at ``p_bytes`` per element. So the bytes a single token adds is,
from first principles:

    bytes_per_token = 2 (K,V) * n_layers * d_model * p_bytes

With the specs below that is 2 * 4 * 512 * 2 = 8192 B = 8 KiB per token, and the
GPU holds 1 GiB = 2**30 B. The GPU therefore fills after exactly

    2**30 / 8192 = 131072 tokens

so utilization == (tokens resident) / 131072. Every expected value here is that
hand-derived fraction, not a snapshot of the function's output.
"""

from hypothesis import given
from hypothesis import strategies as st

from kavier.sdk.inference.stages.kv_usage import get_kv_cache_utilization
from kavier.sdk.library.specs.GPUSpec import GPUSpec
from kavier.sdk.library.specs.LLMSpec import LLMSpec

# 1 GiB of memory -> 2**30 bytes total.
GPU = GPUSpec(
    gpu_name="ToyGPU",
    memory_bandwidth_gbps=600,
    fp_16_tensor_core_tflops=125,
    gpu_cores=9216,
    gpu_core_max_mhz=1695,
    memory_gb=1,
    base_power_w=150,
)
# bytes_per_token = 2 * n_layers(4) * d_model(512) * p_bytes(2) = 8192 B.
LLM = LLMSpec(
    llm_name="Toy",
    n_layers=4,
    d_model=512,
    p_bytes=2,
    m_params=1e6,
    n_heads=8,
    d_head=64,
)

# Tokens needed to fill the whole GPU: 2**30 / 8192.
TOKENS_TO_FILL = 2**30 // 8192  # == 131072
assert TOKENS_TO_FILL == 131072

# Timeline used across the exact-value tests.
T_PREFILL = 2.0
T_DECODE = 4.0
PROMPT_LEN = 96
RESPONSE_LEN = 32


def util(t, **kw):
    params = dict(
        llm=LLM,
        gpu=GPU,
        t_prefill=T_PREFILL,
        t_decode=T_DECODE,
        t=t,
        prompt_len=PROMPT_LEN,
        response_len=RESPONSE_LEN,
        kv_cache=True,
    )
    params.update(kw)
    return get_kv_cache_utilization(**params)


def test_off_returns_exactly_zero():
    # Spec: with the KV cache disabled the model reports no KV memory at all,
    # regardless of how far into the request we are.
    # Falsify: if the kv_cache=False branch were dropped it would return the
    # positive resident fraction instead of 0.
    assert util(t=T_PREFILL + T_DECODE, kv_cache=False) == 0


def test_prefill_ramp_is_linear_from_zero_to_full_prompt():
    # During prefill the resident token count ramps LINEARLY 0 -> prompt_len.
    # t=0        -> 0 tokens
    # t=T/2      -> 48 tokens  (a quadratic ramp would give 96*0.25 = 24, so the
    #               midpoint value falsifies any non-linear ramp)
    # t=T_prefill-> 96 tokens
    assert util(t=0.0) == 0.0
    assert util(t=T_PREFILL / 2) == 48 / TOKENS_TO_FILL
    assert util(t=T_PREFILL) == PROMPT_LEN / TOKENS_TO_FILL  # 96/131072


def test_decode_adds_response_tokens_linearly():
    # After prefill the prompt is fully resident and the response ramps in.
    # Halfway through decode: 96 prompt + 32*0.5 = 112 tokens.
    # Falsify: dropping the prompt_len base, or the response ramp, changes 112.
    t = T_PREFILL + T_DECODE / 2
    assert util(t=t) == 112 / TOKENS_TO_FILL


def test_saturates_and_clamps_at_prompt_plus_response():
    # At the end of decode all prompt+response tokens are resident (96+32=128),
    # and the min(t - t_prefill, t_decode) clamp keeps it flat forever after.
    full = (PROMPT_LEN + RESPONSE_LEN) / TOKENS_TO_FILL  # 128/131072 = 0.0009765625
    assert util(t=T_PREFILL + T_DECODE) == full
    # Falsify the clamp: without min(), t far past decode would keep adding
    # response tokens and utilization would exceed `full`.
    assert util(t=1000.0) == full


def test_zero_prefill_time_makes_prompt_instantly_resident():
    # Edge: t_prefill == 0. At t=0 the whole prompt is already resident
    # (the guard avoids a 0/0), so utilization jumps straight to the prompt
    # fraction rather than starting at zero.
    # Falsify: removing the `t_prefill > 0` guard raises ZeroDivisionError.
    assert util(t=0.0, t_prefill=0.0) == PROMPT_LEN / TOKENS_TO_FILL


def test_zero_decode_time_adds_no_response():
    # Edge: t_decode == 0. Past prefill, only the prompt is resident; the
    # response contributes nothing (the guard avoids a 0/0).
    # Falsify: removing the `t_decode > 0` guard raises ZeroDivisionError;
    # a bug that still added response tokens would give 128/131072 not 96/131072.
    assert util(t=5.0, t_decode=0.0) == PROMPT_LEN / TOKENS_TO_FILL


def test_bytes_per_token_uses_both_kv_tensors():
    # Cross-check the bytes-per-token coefficient against an independent build.
    # Doubling n_layers must exactly double utilization (linear in layer count),
    # because KV bytes scale linearly with the number of layers.
    llm_2x = LLMSpec(
        llm_name="Toy2x",
        n_layers=2 * LLM.n_layers,
        d_model=LLM.d_model,
        p_bytes=LLM.p_bytes,
        m_params=1e6,
        n_heads=8,
        d_head=64,
    )
    t = T_PREFILL + T_DECODE  # fully resident
    base = util(t=t)
    doubled = util(t=t, llm=llm_2x)
    assert doubled == 2 * base


# ---- properties over the whole valid input range ----


@given(
    t1=st.floats(min_value=0.0, max_value=50.0),
    delta=st.floats(min_value=0.0, max_value=50.0),
    t_prefill=st.floats(min_value=0.01, max_value=10.0),
    t_decode=st.floats(min_value=0.01, max_value=10.0),
    prompt_len=st.integers(min_value=0, max_value=10_000),
    response_len=st.integers(min_value=0, max_value=10_000),
)
def test_utilization_is_monotonic_nondecreasing_in_time(t1, delta, t_prefill, t_decode, prompt_len, response_len):
    # Invariant: resident KV bytes never shrink while a request runs, so
    # utilization is non-decreasing in elapsed time. Tokens only accumulate
    # (prefill fills the prompt, decode fills the response) and then clamp.
    t2 = t1 + delta
    u1 = util(t=t1, t_prefill=t_prefill, t_decode=t_decode, prompt_len=prompt_len, response_len=response_len)
    u2 = util(t=t2, t_prefill=t_prefill, t_decode=t_decode, prompt_len=prompt_len, response_len=response_len)
    assert u1 >= 0
    assert u2 >= u1 - 1e-12
