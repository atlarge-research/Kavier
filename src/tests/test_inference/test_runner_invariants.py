"""Behavioural invariants for kavier_inference.core.runner.simulate_one.

test_runner.py pins the ms-conversion and tiling for fixed inputs. This adds the
scaling/monotonicity laws and output sanity that must hold for any request:
  * fragments ALWAYS tile the task duration exactly (regression guard, generalised);
  * longer requests -> longer (never shorter) task duration;
  * KV cache on yields duration <= KV off for the same request;
  * gpu_usage on every fragment is within [0, gpu_capacity];
  * total_tokens always equals n_in + n_out.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from kavier_inference.core.cache import PrefixCache
from kavier_inference.core.config import CacheCfg, SimConfig
from kavier_inference.core.runner import simulate_one
from kavier_library.gpu import GPU_SPEC_LIBRARY
from kavier_library.llm import LLM_SPEC_LIBRARY

_LLM = LLM_SPEC_LIBRARY["Llama-3-8B"]
_GPU = GPU_SPEC_LIBRARY["A10"]


def _run(n_in: int, n_out: int, kv: bool = True):
    cfg = SimConfig(kv_cache=kv, cache=CacheCfg())
    cache = PrefixCache(cfg.cache)
    return simulate_one(
        idx=0, session_id="s", n_in_tokens=n_in, n_out_tokens=n_out, in_tokens=None,
        llm=_LLM, gpu=_GPU, cache=cache, cfg=cfg, export_rate_s=cfg.export_rate, t0_ms=0,
    )


@given(
    n_in=st.integers(min_value=0, max_value=8000),
    n_out=st.integers(min_value=0, max_value=2000),
)
@settings(max_examples=80, deadline=None)
def test_fragments_always_tile_task_duration(n_in, n_out) -> None:
    task, fragments, _tp, _td = _run(n_in, n_out)
    assert task["duration"] >= 1
    assert sum(f["duration"] for f in fragments) == task["duration"]
    assert len(fragments) >= 1


@given(
    n_in=st.integers(min_value=0, max_value=8000),
    n_out=st.integers(min_value=0, max_value=2000),
)
@settings(max_examples=60, deadline=None)
def test_total_tokens_is_sum(n_in, n_out) -> None:
    task, _frags, _tp, _td = _run(n_in, n_out)
    assert task["total_tokens"] == n_in + n_out


def test_longer_request_not_shorter_duration() -> None:
    short = _run(128, 32)[0]["duration"]
    long = _run(2048, 512)[0]["duration"]
    assert long >= short


def test_kv_cache_on_not_slower_than_off() -> None:
    on = _run(512, 256, kv=True)[0]["duration"]
    off = _run(512, 256, kv=False)[0]["duration"]
    assert on <= off


@given(
    n_in=st.integers(min_value=1, max_value=4000),
    n_out=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=60, deadline=None)
def test_fragment_gpu_usage_within_capacity(n_in, n_out) -> None:
    _task, fragments, _tp, _td = _run(n_in, n_out)
    gpu_capacity = float(_GPU.core_max_mhz * _GPU.cores)
    for f in fragments:
        assert 0.0 <= f["gpu_usage"] <= gpu_capacity + 1e-6
        assert f["cpu_count"] == 1 and f["gpu_count"] == 1
