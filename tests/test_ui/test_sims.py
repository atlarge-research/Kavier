"""kavier.ui adapter layer: the pure sim wrappers over the real engines (the interactive widgets
need a TTY and are exercised by hand)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kavier.sdk.co2.engine import WS_PER_KWH
from kavier.sdk.library import UnknownSpecError, get_gpu
from kavier.ui import sims


def _infer_inputs(**over: object) -> dict[str, object]:
    base = {
        "model": "Llama-3-8B",
        "gpu": "A10",
        "num_requests": 32,
        "input_tokens": 256,
        "output_tokens": 64,
        "kv_cache": True,
        "prefix_policy": "prefill",
        "prefix_min_tokens": 1024,
    }
    base.update(over)
    return base


def _train_inputs(**over: object) -> dict[str, object]:
    base = {
        "model": "mistral-7b-v0.1",
        "gpu": "NVIDIA-A100-SXM4-80GB",
        "method": "lora",
        "batch_size": 4,
        "seq_len": 1024,
        "num_gpus": 8,
        "num_nodes": 1,
        "total_tokens": 10_000_000,
    }
    base.update(over)
    return base


def test_name_helpers_return_the_library_keys_sorted() -> None:
    # Contract: model_names/gpu_names are the *sorted* catalog keys. Oracle is the library
    # dict itself sorted independently -- catches an unsorted return or wiring to a wrong dict.
    # (No magic length constant: the catalog may grow; sorted-equality is the real invariant.)
    assert sims.model_names() == sorted(sims.LLM_SPEC_LIBRARY)
    assert sims.gpu_names() == sorted(sims.GPU_SPEC_LIBRARY)
    # sorted() must actually reorder, so the result is not the dict's insertion order by accident
    assert sims.model_names() == sorted(sims.model_names())


def test_run_inference_reports_hand_computed_token_totals() -> None:
    r = sims.run_inference(_infer_inputs(num_requests=32, input_tokens=256, output_tokens=64))
    assert r["model"] == "Llama-3-8B" and r["gpu"] == "A10"
    # 32 requests * (256 prompt + 64 generated) = 10_240 tokens (not input-only, not output-only)
    assert r["total_tokens"] == 10_240


def test_run_inference_total_time_is_prefill_plus_decode() -> None:
    r = sims.run_inference(_infer_inputs())
    # total_s must be the sum of the two phase accumulators (catches dropping a phase)
    assert r["total_s"] == pytest.approx(r["prefill_s"] + r["decode_s"])
    assert r["prefill_s"] > 0 and r["decode_s"] > 0


def test_homogeneous_workload_has_a_flat_latency_distribution() -> None:
    # Single-stream sim of N identical requests -> every request has identical latency,
    # so all percentiles collapse to one value. A per-request-varying bug (e.g. request 0
    # paying a cache-seed cost the rest skip) would spread p50 vs p99.
    r = sims.run_inference(_infer_inputs())
    assert r["p50_ms"] == pytest.approx(r["p95_ms"]) == pytest.approx(r["p99_ms"])
    # and the throughput is exactly total tokens over summed busy time (reciprocal cross-check)
    assert r["throughput_tok_s"] == pytest.approx(r["total_tokens"] / r["total_s"])


def test_below_threshold_prompts_leave_the_prefix_cache_inert() -> None:
    # prefix_policy="prefill" is active, but input_tokens=256 < prefix_min_tokens=1024, so no
    # prompt ever qualifies to be stored -> zero hits. Raising the prompt above the threshold
    # would flip this, so the assert pins the min-length gate, not merely "cache off".
    r = sims.run_inference(_infer_inputs(input_tokens=256, prefix_min_tokens=1024))
    assert r["cache_hits"] == 0
    assert r["cache_hit_ratio"] == 0.0


def test_inference_total_time_scales_linearly_with_request_count() -> None:
    # No batching/queueing: N identical requests cost N * (one request). Doubling the count
    # must exactly double total busy time. A bug summing per request incorrectly (or sharing
    # work across requests) would break the 2x law. prefix_policy="none" keeps every request
    # truly identical (no seed/hit asymmetry).
    common = dict(prefix_policy="none", input_tokens=256, output_tokens=64)
    r1 = sims.run_inference(_infer_inputs(num_requests=16, **common))
    r2 = sims.run_inference(_infer_inputs(num_requests=32, **common))
    assert r2["total_s"] == pytest.approx(2.0 * r1["total_s"])


def test_inference_carbon_bills_gpu_tdp_over_busy_time() -> None:
    r = sims.run_inference(_infer_inputs())
    c = sims.run_carbon_from_inference(r, intensity_g_kwh=400.0)
    # Independent energy oracle (bypasses compute_emissions): inference bills the GPU's max
    # power (A10 TDP = 150 W) over total_s, converted at 1 kWh = 3.6e6 W*s.
    a10_tdp_w = get_gpu("A10").max_power_w
    assert a10_tdp_w == 150.0  # anchors the oracle to the catalog spec
    expected_kwh = a10_tdp_w * r["total_s"] / WS_PER_KWH
    assert c["total_energy_kwh"] == pytest.approx(expected_kwh)
    # flat 400 gCO2/kWh trace -> emissions are exactly energy * intensity
    assert c["total_co2_g"] == pytest.approx(c["total_energy_kwh"] * 400.0)
    assert c["total_co2_kg"] == pytest.approx(c["total_co2_g"] / 1000.0)
    assert c["source"] == "inference"


def test_energy_chain_units_and_price_branch() -> None:
    r = sims.run_inference(_infer_inputs())
    e = sims.energy_from_inference(r, gpu_hour_price=2.5)
    # Wh <-> kWh is a factor of 1000 (catches a /100 or missing conversion)
    assert e["energy_kwh"] == pytest.approx(e["energy_wh"] / 1000.0)
    # tokens/Wh is the exact reciprocal of Wh/token
    assert e["tokens_per_wh"] == pytest.approx(e["total_tokens"] / e["energy_wh"])
    # $/Mtoken = gpu_hours * price * 1e6/tokens, hand-derived from the returned intermediates
    per_m = 1_000_000.0 / e["total_tokens"]
    assert e["financial_per_mtoken"] == pytest.approx(e["gpu_hours"] * 2.5 * per_m)
    # no price -> the financial column is omitted entirely (None), not 0
    assert sims.energy_from_inference(r, gpu_hour_price=None)["financial_per_mtoken"] is None


def test_run_training_merges_full_and_step_metrics() -> None:
    r = sims.run_training(_train_inputs())
    # the aggregate (full) throughput must equal the per-step throughput it was extrapolated from
    assert r["train_tokens_per_second"] == pytest.approx(r["tokens_per_second"])
    # total GPUs = per-node gpus * nodes = 8 * 1
    assert r["total_gpus"] == 8
    # aggregate power is per-GPU power times the GPU count
    assert r["aggregate_power_w"] == pytest.approx(r["gpu_power_watts"] * 8)
    # runtime is exactly job tokens / throughput (independent of the engine's own runtime calc term)
    assert r["train_runtime"] == pytest.approx(10_000_000 / r["train_tokens_per_second"])


def test_training_compute_utilization_is_a_bounded_percentage() -> None:
    # MFU is reported in percent and is a physical fraction, so it must sit in (0, 100].
    r = sims.run_training(_train_inputs())
    assert 0 < r["gpu_compute_utilization"] <= 100
    assert 0 <= r["gpu_memory_utilization"] <= 100


def test_run_training_epochs_match_total_tokens() -> None:
    # Two independent input paths that must resolve to the SAME job size and runtime:
    # epochs*dataset_tokens = 3 * 5_000_000 = 15_000_000, vs total_tokens=15_000_000 directly.
    by_epochs = sims.run_training(_train_inputs(total_tokens=None, epochs=3, dataset_tokens=5_000_000))
    by_total = sims.run_training(_train_inputs(total_tokens=15_000_000))
    assert by_epochs["total_tokens"] == 15_000_000
    assert by_epochs["train_runtime"] == pytest.approx(by_total["train_runtime"])


def test_run_training_without_size_has_no_runtime() -> None:
    # No job size -> runtime is undefined and reported as exactly 0.0 (not NaN, not a guess)
    r = sims.run_training(_train_inputs(total_tokens=None))
    assert r["total_tokens"] is None
    assert r["train_runtime"] == 0.0


def test_carbon_from_training_bills_aggregate_power_over_runtime() -> None:
    c = sims.run_carbon_from_training(_train_inputs(intensity=300.0))
    # Independent energy oracle: training bills aggregate (calibrated) power over runtime at 3.6e6 W*s/kWh.
    expected_kwh = c["power_w"] * c["runtime_s"] / WS_PER_KWH
    assert c["total_energy_kwh"] == pytest.approx(expected_kwh)
    # flat 300 gCO2/kWh -> emissions == energy * 300 exactly
    assert c["total_co2_g"] == pytest.approx(c["total_energy_kwh"] * 300.0)
    assert c["source"] == "training"


def test_carbon_from_training_requires_a_runtime() -> None:
    # total_tokens=0 -> runtime 0 -> cannot bill carbon; the wrapper must reject, not divide by zero
    with pytest.raises(ValueError):
        sims.run_carbon_from_training(_train_inputs(total_tokens=0, intensity=400.0))


def test_export_opendc_writes_one_task_and_fragment_per_request(tmp_path: Path) -> None:
    r = sims.run_inference(_infer_inputs(num_requests=8))
    out = sims.export_opendc(r, tmp_path / "opendc")
    tasks = pd.read_parquet(out / "tasks.parquet")
    frags = pd.read_parquet(out / "fragments.parquet")
    # one row per simulated request (8), one fragment each
    assert len(tasks) == 8 and len(frags) == 8
    assert "total_tokens" in tasks.columns
    # per-task token totals sum to the whole workload: 8 * (256 + 64) = 2_560
    assert int(tasks["total_tokens"].sum()) == 2_560


@pytest.mark.parametrize(
    "inputs",
    [
        _infer_inputs(model="NoSuchModel"),
        _infer_inputs(gpu="NoSuchGPU"),
    ],
)
def test_unknown_spec_propagates(inputs: dict[str, object]) -> None:
    # A name absent from the catalog must surface as UnknownSpecError, not a KeyError deep in the engine
    with pytest.raises(UnknownSpecError):
        sims.run_inference(inputs)
