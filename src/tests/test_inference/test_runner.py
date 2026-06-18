"""Regression tests for kavier_inference.core.runner.simulate_one.

Guards the seconds->milliseconds duration fix: the task ``duration`` field is consumed
downstream (the fragments and kavier_energy) as MILLISECONDS, so simulate_one must emit
ms, not raw seconds (a 1000x under-count that also truncated sub-second requests to 0),
and the fragments must tile the task duration exactly.
"""
from kavier_inference.core.cache import PrefixCache
from kavier_inference.core.config import SimConfig
from kavier_inference.core.runner import simulate_one
from kavier_library.gpu import GPU_SPEC_LIBRARY
from kavier_library.llm import LLM_SPEC_LIBRARY


def _run(n_in: int, n_out: int):
    cfg = SimConfig()
    cache = PrefixCache(cfg.cache)
    llm = next(iter(LLM_SPEC_LIBRARY.values()))
    gpu = next(iter(GPU_SPEC_LIBRARY.values()))
    return simulate_one(
        idx=0, session_id="s", n_in_tokens=n_in, n_out_tokens=n_out, in_tokens=None,
        llm=llm, gpu=gpu, cache=cache, cfg=cfg, export_rate_s=cfg.export_rate, t0_ms=0,
    )


def test_task_duration_is_milliseconds():
    task, _frags, t_prefill, t_decode = _run(512, 128)
    expected_ms = max(1, int(round((t_prefill + t_decode) * 1000)))
    assert task["duration"] == expected_ms
    # proves it's ms, not the old raw-seconds (which would equal the truncated value)
    assert task["duration"] > int(t_prefill + t_decode)


def test_fragments_tile_task_duration():
    task, fragments, _t_prefill, _t_decode = _run(512, 128)
    assert sum(f["duration"] for f in fragments) == task["duration"]


def test_subsecond_request_not_truncated_to_zero():
    task, fragments, _t_prefill, _t_decode = _run(1, 1)
    assert task["duration"] >= 1
    assert sum(f["duration"] for f in fragments) == task["duration"]
