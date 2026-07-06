"""Behavioural contracts for ``simulate_one`` (per-request inference sim).

Oracles are hand-derived from the roofline physics and the pinned catalog constants, NOT
from the engine's own output. Reference numbers for Llama-3-8B on an A10:

  active_params = 8e9, p_bytes = 2
  A10: fp16 tensor TFLOPS = 125, mem bandwidth = 600e9 B/s, cores = 9216, core_max = 1695 MHz
  COMPUTE_EFFICIENCY = 0.30, MEMORY_EFFICIENCY = 0.60, PREFILL_OVERHEAD_S = 0.025
  MAX_GPU_UTILIZATION = 0.95, warm = cool = 0.2 s

  per-token decode cost = max(compute, memory):
    compute = 2*8e9 / (125e12*0.30)          = 4.26667e-4 s
    memory  = (2*8e9) / (600e9*0.60)          = 4.44444e-2 s   <- memory-bound wins
  per-input-token prefill cost = 2*8e9 / (125e12*0.30) = 4.26667e-4 s
  gpu_capacity = 1695 * 9216 = 15_621_120
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kavier.sdk.inference.core.cache import PrefixCache
from kavier.sdk.inference.core.config import CacheCfg, SimConfig
from kavier.sdk.inference.core.runner import simulate_one
from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY
from kavier.sdk.library.llm import LLM_SPEC_LIBRARY

_LLM = LLM_SPEC_LIBRARY["Llama-3-8B"]
_GPU = GPU_SPEC_LIBRARY["A10"]

_GPU_CAPACITY = 15_621_120.0  # 1695 MHz * 9216 cores, independent of the engine


def _run(n_in, n_out, kv=True, cache=None, cfg=None, in_tokens=None, sid="s"):
    cfg = cfg or SimConfig(kv_cache=kv, cache=CacheCfg())
    cache = cache if cache is not None else PrefixCache(cfg.cache)
    return simulate_one(
        idx=0,
        session_id=sid,
        n_in_tokens=n_in,
        n_out_tokens=n_out,
        in_tokens=in_tokens,
        llm=_LLM,
        gpu=_GPU,
        cache=cache,
        cfg=cfg,
        export_rate_s=cfg.export_rate,
        t0_ms=0,
    )


@given(
    n_in=st.integers(min_value=0, max_value=8000),
    n_out=st.integers(min_value=0, max_value=2000),
)
@settings(max_examples=80, deadline=None)
def test_total_tokens_is_input_plus_output(n_in, n_out) -> None:
    # Contract: task carries n_in + n_out (the per-token energy step relies on it).
    # Independent oracle: the sum itself. Falsifies under n_in*n_out, n_in only, etc.
    task, *_ = _run(n_in, n_out)
    assert task["total_tokens"] == n_in + n_out


def test_duration_matches_hand_derived_roofline_latency() -> None:
    # n_in=1000, n_out=100, KV on. End-to-end latency through prefill + memory-bound decode.
    #   t_prefill = 0.025 + 1000 * 4.26667e-4 = 0.45166667 s
    #   t_decode  = 100 * 0.04444444          = 4.44444444 s   (memory-bound)
    #   total     = 4.89611111 s -> round(4896.111) = 4896 ms
    # Falsifies if the 1e12 TFLOPS scale, the efficiencies, the roofline max, or the
    # 0.025 prefill overhead are perturbed (e.g. dropping the overhead gives 4871).
    task, *_ = _run(1000, 100, kv=True)
    assert task["duration"] == 4896


def test_kv_off_applies_quadratic_decode_scaling() -> None:
    # n_in=0, n_out=10. Prefill is just the 0.025 s overhead in both cases; only decode differs.
    #   KV on : t_decode = 10        * 0.04444444 = 0.44444 s -> total 0.46944 -> 469 ms
    #   KV off: t_decode = 10*11/2   * 0.04444444 = 2.44444 s -> total 2.46944 -> 2469 ms
    # The n(n+1)/2 = 55 vs 10 factor is the quadratic-without-KV law. Falsifies if the KV
    # branch is ignored (off would equal on) or the triangular sum is dropped.
    on = _run(0, 10, kv=True)[0]["duration"]
    off = _run(0, 10, kv=False)[0]["duration"]
    assert on == 469
    assert off == 2469


@given(
    n_in=st.integers(min_value=0, max_value=8000),
    n_out=st.integers(min_value=0, max_value=2000),
)
@settings(max_examples=80, deadline=None)
def test_fragments_tile_task_duration_exactly(n_in, n_out) -> None:
    # Invariant: the emitted GPU-usage fragments partition the task duration with no gap
    # or overlap. Falsifies if the final fragment stops absorbing the residual, or if
    # fragment durations are computed from a rate inconsistent with the task duration.
    task, fragments, _tp, _td = _run(n_in, n_out)
    assert sum(f["duration"] for f in fragments) == task["duration"]


def test_gpu_usage_takes_only_the_two_piecewise_levels() -> None:
    # The piecewise model yields exactly 0.5 (warm/cool) and MAX_GPU_UTILIZATION=0.95 (steady),
    # scaled by capacity. For a ~4.9 s request the steady window (t in [0.2, total-0.2]) exists,
    # so both levels appear; the very first fragment (t=0 < warm) must be the 0.5 level.
    #   0.5  * 15_621_120 = 7_810_560
    #   0.95 * 15_621_120 = 14_840_064
    # Falsifies if MAX_GPU_UTILIZATION changes, the warm level changes, or capacity is mis-scaled.
    _task, fragments, _tp, _td = _run(1000, 100)
    warm_level = 7_810_560.0
    steady_level = 14_840_064.0
    assert fragments[0]["gpu_usage"] == warm_level
    assert steady_level in {f["gpu_usage"] for f in fragments}
    assert {f["gpu_usage"] for f in fragments} == {warm_level, steady_level}


@pytest.mark.parametrize(
    ("action", "expected_second_duration"),
    [
        # A repeated >=min_len prompt hits the prefix cache on the 2nd request. The hit
        # zeroes stages per the action; the 1st request always pays full price (817 ms:
        # t_prefill = 0.025 + 1024*4.26667e-4 = 0.461907 s, t_decode = 8*0.04444444 = 0.355556 s
        # -> 0.817462 s -> 817 ms).
        ("none", 817),  # no zeroing even on a hit -> same as first request
        ("prefill", 356),  # t_prefill -> 0, decode stays: round(355.556) = 356 ms
        ("full", 1),  # both stages -> 0 s -> max(1, round(0)) = 1 ms (duration floor)
    ],
)
def test_prefix_cache_hit_zeroes_stages_per_action(action, expected_second_duration) -> None:
    cfg = SimConfig(kv_cache=True, cache=CacheCfg(action=action, min_len=1024))
    cache = PrefixCache(cfg.cache)
    tokens = list(range(1024))
    first = _run(1024, 8, cache=cache, cfg=cfg, in_tokens=tokens)[0]["duration"]
    second = _run(1024, 8, cache=cache, cfg=cfg, in_tokens=tokens)[0]["duration"]
    # First request is a cache miss (insert), so it always pays the full 817 ms.
    assert first == 817
    assert second == expected_second_duration
