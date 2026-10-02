"""Tests for ``simulate_one``: one request -> OpenDC task + GPU-usage fragments.

Expected values are computed by hand from the constants and the A10 / Llama-3-8B specs:

  COMPUTE_EFFICIENCY=0.30  MEMORY_EFFICIENCY=0.60  PREFILL_OVERHEAD_S=0.025  MAX_GPU_UTILIZATION=0.95
  Llama-3-8B: active_params=8e9, p_bytes=2
  A10:        fp16_tflops=125, mem_bw=600 GB/s, cores=9216, core_max_mhz=1695

  f_gpu        = 125e12 * 0.30                      = 3.75e13 FLOP/s
  prefill(n)   = 0.025 + n * (2*8e9) / 3.75e13      s
  per_tok      = max(2*8e9/3.75e13, 2*8e9/(600e9*0.60))
               = max(4.2667e-4, 4.4444e-2) = 4.4444e-2 s   (memory-bound)
  decode(n)    = n * per_tok                        s  (kv_cache on -> linear)
  duration_ms  = round((prefill+decode) * 1000), floored at 1
"""

import pytest

from kavier.sdk.inference.core.cache import PrefixCache
from kavier.sdk.inference.core.config import SimConfig
from kavier.sdk.inference.core.runner import simulate_one
from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY
from kavier.sdk.library.llm import LLM_SPEC_LIBRARY

LLM = LLM_SPEC_LIBRARY["Llama-3-8B"]
GPU = GPU_SPEC_LIBRARY["A10"]

# A10 fragment GPU capacity = core_max_mhz * cores = 1695 * 9216.
A10_CAPACITY = 1695 * 9216  # 15_621_120


def _run(n_in, n_out, *, cache=None, in_tokens=None, session_id="s"):
    cfg = SimConfig()
    cache = cache if cache is not None else PrefixCache(cfg.cache)
    return simulate_one(
        idx=0,
        session_id=session_id,
        n_in_tokens=n_in,
        n_out_tokens=n_out,
        in_tokens=in_tokens,
        llm=LLM,
        gpu=GPU,
        cache=cache,
        cfg=cfg,
        export_rate_s=cfg.export_rate,
        t0_ms=0,
    )


def test_task_duration_is_hand_derived_milliseconds():
    # prefill(512) = 0.025 + 512*(1.6e10/3.75e13)   = 0.24345333 s
    # decode(128)  = 128 * 4.4444e-2                = 5.68888889 s
    # total        = 5.93234222 s -> round(*1000)   = 5932 ms
    # int(total_s) would give 5; using the compute term alone would shrink decode ~100x.
    task, _frags, _tp, _td = _run(512, 128)
    assert task["duration"] == 5932


def test_subsecond_request_not_truncated_to_zero():
    # prefill(1)=0.02542667 s, decode(1)=0.04444444 s, total=0.06987111 s -> round(*1000)=70 ms.
    # int(total_s) would give 0.
    task, _frags, _tp, _td = _run(1, 1)
    assert task["duration"] == 70


@pytest.mark.parametrize(
    ("n_in", "n_out"),
    [
        (1, 1),  # total 0.070 s -> num_snaps=max(1,0)=1: one fragment holds the whole residual
        (512, 128),  # total 5.93 s -> 59 snaps of 100 ms + a residual last fragment
    ],
)
def test_fragments_tile_task_duration_exactly(n_in, n_out):
    # Fragments partition the task duration; the last fragment absorbs the residual.
    task, fragments, _tp, _td = _run(n_in, n_out)
    assert sum(f["duration"] for f in fragments) == task["duration"]


@pytest.mark.parametrize(
    ("export_rate_s", "n_fragments"),
    [
        (0.0015, 35),  # 1.5 ms rounds to 2 ms fragments: 70 // 2 = 35
        (0.0006, 70),  # 0.6 ms rounds to 1 ms fragments
        (0.1234, 1),  # one 123 ms fragment is longer than the 70 ms task
    ],
)
def test_fragments_tile_task_duration_at_fractional_ms_export_rates(export_rate_s, n_fragments):
    # A 70 ms request (see test_subsecond_request_not_truncated_to_zero).
    cfg = SimConfig(export_rate=export_rate_s)
    task, fragments, _tp, _td = simulate_one(
        idx=0,
        session_id="s",
        n_in_tokens=1,
        n_out_tokens=1,
        in_tokens=None,
        llm=LLM,
        gpu=GPU,
        cache=PrefixCache(cfg.cache),
        cfg=cfg,
        export_rate_s=export_rate_s,
        t0_ms=0,
    )
    assert task["duration"] == 70
    assert sum(f["duration"] for f in fragments) == 70
    assert len(fragments) == n_fragments
    assert all(f["duration"] >= 1 for f in fragments)


def test_task_total_tokens_is_input_plus_output_sum():
    # 40 + 60 = 100. Unequal inputs catch n_in+n_in (80) and n_out+n_out (120).
    task, _frags, _tp, _td = _run(40, 60)
    assert task["total_tokens"] == 100


def test_fragment_gpu_usage_spans_util_endpoints():
    # gpu_usage = utilisation * capacity; utilisation is 0.5 in warm/cool windows and
    # MAX_GPU_UTILIZATION=0.95 in steady state. A multi-second request has both.
    #   max = 0.95 * (1695*9216) = 14_840_064.0
    #   min = 0.50 * (1695*9216) =  7_810_560.0
    _task, fragments, _tp, _td = _run(512, 128)
    usages = [f["gpu_usage"] for f in fragments]
    assert max(usages) == pytest.approx(0.95 * A10_CAPACITY)
    assert min(usages) == pytest.approx(0.50 * A10_CAPACITY)


def test_prefix_cache_hit_zeroes_prefill_time():
    # Default policy "prefill": a prefix hit sets prefill to 0. A hit needs n_in >= min_len (1024)
    # and an earlier lookup in the same session.
    #   miss: prefill(1024)+decode(128) = 0.46190667 + 5.68888889 = 6.1508 s -> 6151 ms
    #   hit : decode(128) only          = 5.68888889 s              -> 5689 ms
    cfg = SimConfig()
    shared_cache = PrefixCache(cfg.cache)
    prompt = list(range(1024))

    miss_task, _f, _tp, _td = _run(1024, 128, cache=shared_cache, in_tokens=prompt)
    hit_task, _f2, _tp2, _td2 = _run(1024, 128, cache=shared_cache, in_tokens=prompt)

    assert miss_task["duration"] == 6151
    assert hit_task["duration"] == 5689
    assert hit_task["duration"] < miss_task["duration"]
