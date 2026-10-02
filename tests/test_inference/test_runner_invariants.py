"""Property and roofline tests for ``simulate_one``, the per-request inference simulation.

Expected values are computed by hand. Reference numbers for Llama-3-8B on an A10:

  active_params = 8e9, p_bytes = 2
  A10: fp16 tensor TFLOPS = 125, mem bandwidth = 600e9 B/s, cores = 9216, core_max = 1695 MHz
  COMPUTE_EFFICIENCY = 0.30, MEMORY_EFFICIENCY = 0.60, PREFILL_OVERHEAD_S = 0.025
  MAX_GPU_UTILIZATION = 0.95, warm = cool = 0.2 s

  per-token decode cost = max(compute, memory):
    compute = 2*8e9 / (125e12*0.30)          = 4.26667e-4 s
    memory  = (2*8e9) / (600e9*0.60)          = 4.44444e-2 s   <- memory-bound
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

_GPU_CAPACITY = 15_621_120.0  # 1695 MHz * 9216 cores


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
    # The per-token energy step reads total_tokens = n_in + n_out.
    task, *_ = _run(n_in, n_out)
    assert task["total_tokens"] == n_in + n_out


def test_duration_matches_hand_derived_roofline_latency() -> None:
    # n_in=1000, n_out=100, KV on.
    #   t_prefill = 0.025 + 1000 * 4.26667e-4 = 0.45166667 s
    #   t_decode  = 100 * 0.04444444          = 4.44444444 s   (memory-bound)
    #   total     = 4.89611111 s -> round(4896.111) = 4896 ms
    # Without the 0.025 s overhead: 4871 ms.
    task, *_ = _run(1000, 100, kv=True)
    assert task["duration"] == 4896


def test_kv_off_applies_quadratic_decode_scaling() -> None:
    # n_in=0, n_out=10. Prefill is the 0.025 s overhead in both cases.
    #   KV on : t_decode = 10        * 0.04444444 = 0.44444 s -> total 0.46944 -> 469 ms
    #   KV off: t_decode = 10*11/2   * 0.04444444 = 2.44444 s -> total 2.46944 -> 2469 ms
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
    # Fragments partition the task duration; the last fragment absorbs the residual.
    task, fragments, _tp, _td = _run(n_in, n_out)
    assert sum(f["duration"] for f in fragments) == task["duration"]


@given(
    n_in=st.integers(min_value=0, max_value=4000),
    n_out=st.integers(min_value=0, max_value=200),
    export_rate=st.floats(min_value=1e-4, max_value=2.0),
)
@settings(max_examples=100, deadline=None)
def test_fragments_tile_task_duration_for_any_export_rate(n_in, n_out, export_rate) -> None:
    task, fragments, _tp, _td = _run(n_in, n_out, cfg=SimConfig(export_rate=export_rate))
    assert sum(f["duration"] for f in fragments) == task["duration"]
    assert all(f["duration"] >= 1 for f in fragments)


def test_gpu_usage_takes_only_the_two_piecewise_levels() -> None:
    # Piecewise utilisation: 0.5 in warm/cool windows, MAX_GPU_UTILIZATION=0.95 in steady state,
    # times capacity. A ~4.9 s request has a steady window t in [0.2, total-0.2]; the first
    # fragment (t=0) is in the warm window.
    #   0.5  * 15_621_120 = 7_810_560
    #   0.95 * 15_621_120 = 14_840_064
    _task, fragments, _tp, _td = _run(1000, 100)
    warm_level = 7_810_560.0
    steady_level = 14_840_064.0
    assert fragments[0]["gpu_usage"] == warm_level
    assert steady_level in {f["gpu_usage"] for f in fragments}
    assert {f["gpu_usage"] for f in fragments} == {warm_level, steady_level}


@pytest.mark.parametrize(
    ("action", "expected_second_duration"),
    [
        # The second identical prompt (>= min_len) is a cache hit; the action sets which stages drop.
        # First request: t_prefill = 0.025 + 1024*4.26667e-4 = 0.461907 s,
        # t_decode = 8*0.04444444 = 0.355556 s -> 0.817462 s -> 817 ms.
        ("none", 817),  # hit changes nothing
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
    # First request is a miss.
    assert first == 817
    assert second == expected_second_duration
