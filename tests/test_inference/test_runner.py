"""Behavioural tests for ``simulate_one`` (one request -> OpenDC task + GPU-usage fragments).

Every expected value below is hand-derived from the physics constants and the pinned A10 /
Llama-3-8B specs, NOT read back from the code under test. The physics (independent of runner.py):

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

# Pin the specs by name so the hand-derived oracles stay valid regardless of catalogue order.
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
    # Falsification: the old raw-seconds bug (int(total_s)) yields 5, /1000 -> 6, dropping the
    # roofline max() (using the compute term) collapses decode ~100x.
    task, _frags, _tp, _td = _run(512, 128)
    assert task["duration"] == 5932


def test_subsecond_request_not_truncated_to_zero():
    # prefill(1)=0.02542667 s, decode(1)=0.04444444 s, total=0.06987111 s -> round(*1000)=70 ms.
    # The old int(total_s) truncated this sub-second request to 0; the ms model yields exactly 70.
    task, _frags, _tp, _td = _run(1, 1)
    assert task["duration"] == 70


@pytest.mark.parametrize(
    ("n_in", "n_out"),
    [
        (1, 1),  # total 0.070 s -> num_snaps=max(1,0)=1: single fragment carries the whole residual
        (512, 128),  # total 5.93 s -> 59 snaps of 100 ms + a residual last fragment
    ],
)
def test_fragments_tile_task_duration_exactly(n_in, n_out):
    # Invariant: the fragments must partition the task duration with no gap/overlap, because the
    # final fragment absorbs the residual. Falsification: emit a fixed fragment_duration for the
    # last fragment too and the sum drifts from task["duration"].
    task, fragments, _tp, _td = _run(n_in, n_out)
    assert sum(f["duration"] for f in fragments) == task["duration"]


def test_task_total_tokens_is_input_plus_output_sum():
    # Independent: 40 + 60 = 100. Asymmetric inputs so n_in+n_in (80) or n_out+n_out (120) fail.
    task, _frags, _tp, _td = _run(40, 60)
    assert task["total_tokens"] == 100


def test_fragment_gpu_usage_spans_util_endpoints():
    # gpu_usage = utilisation * capacity, utilisation is 0.5 in warm/cool windows and
    # MAX_GPU_UTILIZATION=0.95 in steady state. For a multi-second request both endpoints appear.
    #   max = 0.95 * (1695*9216) = 14_840_064.0
    #   min = 0.50 * (1695*9216) =  7_810_560.0
    # Falsification: a wrong capacity formula (e.g. cores only) or a changed util cap shifts these.
    _task, fragments, _tp, _td = _run(512, 128)
    usages = [f["gpu_usage"] for f in fragments]
    assert max(usages) == pytest.approx(0.95 * A10_CAPACITY)
    assert min(usages) == pytest.approx(0.50 * A10_CAPACITY)


def test_prefix_cache_hit_zeroes_prefill_time():
    # Default cache policy is "prefill": a prefix hit drops prefill to 0, leaving decode-only.
    # Requires n_in >= cache.min_len (1024) and a prior seeding lookup on the same session.
    #   miss: prefill(1024)+decode(128) = 0.46190667 + 5.68888889 = 6.1508 s -> 6151 ms
    #   hit : decode(128) only          = 5.68888889 s              -> 5689 ms
    # Falsification: if the hit branch didn't zero prefill, the second run would also be 6151 ms.
    cfg = SimConfig()
    shared_cache = PrefixCache(cfg.cache)
    prompt = list(range(1024))

    miss_task, _f, _tp, _td = _run(1024, 128, cache=shared_cache, in_tokens=prompt)
    hit_task, _f2, _tp2, _td2 = _run(1024, 128, cache=shared_cache, in_tokens=prompt)

    assert miss_task["duration"] == 6151
    assert hit_task["duration"] == 5689
    assert hit_task["duration"] < miss_task["duration"]
