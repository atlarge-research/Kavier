"""Label pins for ``kavier.ui.render`` (zero-coverage module, de-slop Phase 0 Deliverable B).

Every public function in render.py builds a Rich renderable (Panel/Table) from a result dict produced
by ``kavier.ui.sims`` -- the same wiring ``kavier.ui.app`` uses. This module renders each one through
``rich.console.Console(record=True)`` and asserts the metric LABEL strings it prints, so a later
CLI/UI label unification (a de-slop candidate) cannot silently rename/drop one without a red test. It
does not assert formatted VALUES (that's a snapshot of formatting, not a contract) or internal layout.
"""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from kavier.ui import render, sims


def _text(renderable: Any) -> str:
    """Render one Rich renderable to plain text (no ANSI) at a width wide enough to avoid wrapping."""
    console = Console(record=True, width=200)
    console.print(renderable)
    return console.export_text()


# --------------------------------------------------------------------------- spinner


def test_spinner_yields_control_to_its_body() -> None:
    # spinner() is a @contextmanager; its one behaviour worth pinning is that it actually yields
    # control to the body exactly once. A broken context manager (raises before yield, or swallows
    # the body) would leave `executed` False.
    executed = False
    with render.spinner("doing work", "cyan"):
        executed = True
    assert executed


# --------------------------------------------------------------------------- specs_panel


def test_specs_panel_shows_llm_and_gpu_labels() -> None:
    text = _text(render.specs_panel("Llama-3-8B", "A10"))
    assert "selected specs" in text
    # sub-panel titles are the spec names themselves.
    assert "Llama-3-8B" in text and "A10" in text
    for label in ("params", "active", "layers", "d_model", "heads", "bytes/param"):
        assert label in text
    for label in ("memory", "FP16 TC", "bandwidth", "max power", "MFU factor", "calib"):
        assert label in text


# --------------------------------------------------------------------------- inference_result


def _inference_dict() -> dict[str, Any]:
    return sims.run_inference(
        {
            "model": "Llama-3-8B",
            "gpu": "A10",
            "num_requests": 8,
            "input_tokens": 64,
            "output_tokens": 32,
            "kv_cache": True,
            "prefix_policy": "prefill",
            "prefix_min_tokens": 1024,
        }
    )


def test_inference_result_shows_all_metric_labels() -> None:
    text = _text(render.inference_result(_inference_dict()))
    assert "Inference" in text and "Inference results" in text
    for label in ("requests", "in/out tokens", "KV cache", "prefix policy"):
        assert label in text
    for label in (
        "Prefill time",
        "Decode time",
        "Total time",
        "Throughput",  # appears twice: tok/s row and req/s row
        "Mean TTFT",
        "Latency p50 / p95 / p99",
        "Cache hit ratio",
        "LRU evictions",
    ):
        assert label in text


# --------------------------------------------------------------------------- training_result


def _training_dict(**over: Any) -> dict[str, Any]:
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
    return sims.run_training(base)


def test_training_result_shows_core_metric_labels() -> None:
    text = _text(render.training_result(_training_dict()))
    assert "Training" in text and "Training results" in text
    for label in ("method", "batch × seq", "GPUs", "total tokens"):
        assert label in text
    for label in (
        "Throughput",
        "Per-GPU throughput",
        "Samples/s",
        "Steps/s",
        "Step time",
        "MFU",
        "Memory util",
        "Power / GPU",
        "Aggregate power",
    ):
        assert label in text


@pytest.mark.parametrize(
    ("total_tokens", "expect_runtime_row"),
    [
        (10_000_000, True),  # a sized job -> render.py appends the "Training runtime" row
        (None, False),  # no job size -> the row is conditionally omitted (r["total_tokens"] falsy)
    ],
)
def test_training_result_runtime_row_is_conditional_on_total_tokens(
    total_tokens: int | None, expect_runtime_row: bool
) -> None:
    text = _text(render.training_result(_training_dict(total_tokens=total_tokens)))
    assert ("Training runtime" in text) is expect_runtime_row


# --------------------------------------------------------------------------- energy_result


def test_energy_result_shows_metric_labels() -> None:
    e = sims.energy_from_inference(_inference_dict(), gpu_hour_price=2.5)
    text = _text(render.energy_result(e))
    assert "Energy" in text
    for label in (
        "Total energy",  # appears twice: Wh row and kWh row
        "Energy efficiency",
        "Carbon efficiency",
        "Cost efficiency",
        "Tokens / Wh",
        "GPU-hours",
    ):
        assert label in text


def test_energy_result_cost_is_na_without_a_price() -> None:
    # energy_result formats financial_per_mtoken as "N/A" when the verb was given no gpu_hour_price
    # (energy_from_inference returns financial_per_mtoken=None, not a guessed default).
    e = sims.energy_from_inference(_inference_dict(), gpu_hour_price=None)
    assert e["financial_per_mtoken"] is None  # precondition for the branch under test
    text = _text(render.energy_result(e))
    assert "N/A" in text


# --------------------------------------------------------------------------- carbon_result


def test_carbon_result_shows_metric_labels_and_per_mtoken_row_when_tokens_present() -> None:
    c = sims.run_carbon_from_inference(_inference_dict(), intensity_g_kwh=400.0)
    assert c["total_tokens"] > 0  # precondition: the conditional row below must be shown
    text = _text(render.carbon_result(c))
    assert "Carbon emissions" in text
    for label in ("Source", "Carbon intensity", "Run power", "Runtime", "Energy", "Emissions"):
        assert label in text
    assert "Carbon / Mtoken" in text


def test_carbon_result_omits_per_mtoken_row_when_tokens_falsy() -> None:
    # Hand-built minimal dict (not a real sims run): carbon_result's own guard is `if r["total_tokens"]:`
    # -- every OTHER test in this file exercises that dict shape via real physics, so this one isolates
    # the branch itself with total_tokens=0, which run_carbon_from_inference/_training cannot produce
    # (the latter raises before returning when the job has no size; the former needs 0-token requests).
    fake = {
        "source": "test",
        "intensity": 400.0,
        "power_w": 150.0,
        "runtime_s": 10.0,
        "total_energy_kwh": 0.0004,
        "total_co2_g": 0.16,
        "total_co2_kg": 0.00016,
        "total_tokens": 0,
    }
    text = _text(render.carbon_result(fake))
    assert "Carbon emissions" in text  # the panel itself still renders
    assert "Carbon / Mtoken" not in text
