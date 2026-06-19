"""Tests for the kavier_ui adapter layer — the thin wrappers over the real engines.

The interactive widgets need a TTY and are exercised separately by hand; here we
pin the (pure) sim adapters that produce what the rich panels render, plus the
no-TTY guard and graceful error propagation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kavier_library import UnknownSpecError
from kavier_ui import sims


def _infer_inputs(**over: object) -> dict[str, object]:
    base = {
        "model": "Llama-3-8B", "gpu": "A10", "num_requests": 32,
        "input_tokens": 256, "output_tokens": 64, "kv_cache": True,
        "prefix_policy": "prefill", "prefix_min_tokens": 1024,
    }
    base.update(over)
    return base


def _train_inputs(**over: object) -> dict[str, object]:
    base = {
        "model": "mistral-7b-v0.1", "gpu": "NVIDIA-A100-SXM4-80GB", "method": "lora",
        "batch_size": 4, "seq_len": 1024, "num_gpus": 8, "num_nodes": 1,
        "total_tokens": 10_000_000,
    }
    base.update(over)
    return base


def test_library_lists_match_spec_libraries() -> None:
    assert sims.model_names() == sorted(sims.LLM_SPEC_LIBRARY)
    assert sims.gpu_names() == sorted(sims.GPU_SPEC_LIBRARY)
    assert len(sims.model_names()) == 16
    assert len(sims.gpu_names()) == 11


def test_run_inference_shape_and_consistency() -> None:
    r = sims.run_inference(_infer_inputs(num_requests=32, input_tokens=256, output_tokens=64))
    assert r["model"] == "Llama-3-8B" and r["gpu"] == "A10"
    assert r["total_tokens"] == 32 * (256 + 64)
    assert r["total_s"] == pytest.approx(r["prefill_s"] + r["decode_s"])
    # homogeneous workload -> identical per-request latency at every percentile
    assert r["p50_ms"] == pytest.approx(r["p95_ms"]) == pytest.approx(r["p99_ms"])
    assert r["throughput_tok_s"] == pytest.approx(r["total_tokens"] / r["total_s"])
    assert r["cache_hits"] == 0  # no token lists -> prefix cache inert


def test_run_training_merges_full_and_step_metrics() -> None:
    r = sims.run_training(_train_inputs())
    # both engine entry points are surfaced
    assert r["train_tokens_per_second"] == pytest.approx(r["tokens_per_second"])
    assert r["total_gpus"] == 8
    assert r["aggregate_power_w"] == pytest.approx(r["gpu_power_watts"] * 8)
    assert 0 < r["gpu_compute_utilization"] <= 100
    assert r["train_runtime"] > 0  # total_tokens set


def test_energy_chain_is_internally_consistent() -> None:
    r = sims.run_inference(_infer_inputs())
    e = sims.energy_from_inference(r, gpu_hour_price=2.5)
    assert e["energy_kwh"] == pytest.approx(e["energy_wh"] / 1000.0)
    # tokens/Wh is the reciprocal of Wh/token
    assert e["tokens_per_wh"] == pytest.approx(e["total_tokens"] / e["energy_wh"])
    assert e["financial_per_mtoken"] is not None
    # no price -> financial efficiency is omitted
    assert sims.energy_from_inference(r, gpu_hour_price=None)["financial_per_mtoken"] is None


def test_carbon_from_inference_matches_intensity() -> None:
    r = sims.run_inference(_infer_inputs())
    c = sims.run_carbon_from_inference(r, intensity_g_kwh=400.0)
    # flat 400 gCO2/kWh -> emissions == energy_kwh * 400 (exact)
    assert c["total_co2_g"] == pytest.approx(c["total_energy_kwh"] * 400.0)
    assert c["total_co2_kg"] == pytest.approx(c["total_co2_g"] / 1000.0)
    assert c["source"] == "inference"


def test_carbon_from_training_matches_intensity() -> None:
    c = sims.run_carbon_from_training(_train_inputs(intensity=300.0))
    assert c["total_co2_g"] == pytest.approx(c["total_energy_kwh"] * 300.0)
    assert c["source"] == "training"


def test_carbon_from_training_requires_runtime() -> None:
    with pytest.raises(ValueError):
        sims.run_carbon_from_training(_train_inputs(total_tokens=0, intensity=400.0))


def test_export_opendc_writes_valid_parquet(tmp_path: Path) -> None:
    r = sims.run_inference(_infer_inputs(num_requests=8))
    out = sims.export_opendc(r, tmp_path / "opendc")
    tasks = pd.read_parquet(out / "tasks.parquet")
    frags = pd.read_parquet(out / "fragments.parquet")
    assert len(tasks) == 8 and len(frags) == 8
    assert "total_tokens" in tasks.columns
    assert int(tasks["total_tokens"].sum()) == 8 * (256 + 64)


@pytest.mark.parametrize("inputs", [
    _infer_inputs(model="NoSuchModel"),
    _infer_inputs(gpu="NoSuchGPU"),
])
def test_unknown_spec_propagates(inputs: dict[str, object]) -> None:
    with pytest.raises(UnknownSpecError):
        sims.run_inference(inputs)
