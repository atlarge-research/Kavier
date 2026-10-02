"""Tests for the public API shown in ``docs/usage.py``.

One test runs the script. The others recompute each predicted column of the ``inference`` and
``training`` verbs from arithmetic, physics, library constants, invariants, or scaling laws.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import kavier
from kavier.sdk.library import get_gpu
from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY

USAGE = Path(__file__).resolve().parents[2] / "docs" / "usage.py"

# The two workloads from docs/usage.py.
INFER = {"model": "Llama-3-8B", "gpu": "A10", "num_requests": 128, "input_tokens": 512, "output_tokens": 128}
TRAIN_LORA = {
    "model": "mistral-7b-v0.1",
    "gpu": "NVIDIA-A100-SXM4-80GB",
    "method": "lora",
    "batch_size": 4,
    "seq_len": 1024,
    "num_gpus": 8,
    "num_nodes": 1,
    "epochs": 3,
    "dataset_tokens": 5_000_000,
}


# --------------------------------------------------------------------------- #
# Integration: docs/usage.py runs.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not USAGE.exists(), reason="docs/usage.py missing")
def test_usage_example_script_runs_and_emits_both_tables() -> None:
    # pandas hides middle columns behind "...", so check edge columns that always print:
    # total_tokens (inference table) and gpu_power_watts (training table).
    proc = subprocess.run([sys.executable, str(USAGE)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "total_tokens" in proc.stdout  # inference performance table printed
    assert "gpu_power_watts" in proc.stdout  # training performance table printed


# --------------------------------------------------------------------------- #
# Inference verbs
# --------------------------------------------------------------------------- #
def test_inference_total_tokens_is_requests_times_prompt_plus_output() -> None:
    perf = kavier.inference.performance(INFER).iloc[0]
    # 128 requests x (512 in + 128 out) = 128 * 640 = 81_920.
    assert perf["total_tokens"] == 128 * (512 + 128) == 81_920


def test_identical_requests_give_equal_p50_and_p95() -> None:
    # Single stream: n identical requests have equal latencies, so p50 == p95.
    perf = kavier.inference.performance(INFER).iloc[0]
    assert perf["p50_ms"] == pytest.approx(perf["p95_ms"])


def test_inference_throughput_is_invariant_under_request_count() -> None:
    # No batching or contention: total_s and total_tokens are linear in num_requests,
    # so throughput_tok_s does not depend on it.
    one = kavier.inference.performance({**INFER, "num_requests": 1}).iloc[0]
    many = kavier.inference.performance({**INFER, "num_requests": 256}).iloc[0]
    assert many["throughput_tok_s"] == pytest.approx(one["throughput_tok_s"], rel=1e-9)
    # 256 identical requests take 256x one request.
    assert many["total_s"] == pytest.approx(256 * one["total_s"], rel=1e-9)


def test_inference_energy_bills_gpu_tdp_over_busy_time() -> None:
    # Inference bills GPU max power (TDP): Wh = W * s / 3600. A10 TDP = 150 W.
    perf = kavier.inference.performance(INFER).iloc[0]
    ener = kavier.inference.energy(INFER).iloc[0]
    tdp_w = get_gpu("A10").max_power_w
    assert tdp_w == 150  # catalog value
    expected_wh = tdp_w * perf["total_s"] / 3600.0
    assert ener["energy_wh"] == pytest.approx(expected_wh)
    # kWh = Wh / 1000.
    assert ener["energy_kwh"] == pytest.approx(ener["energy_wh"] / 1000.0)


def test_inference_carbon_from_energy_and_intensity() -> None:
    # gCO2 = energy(kWh) * intensity(gCO2/kWh) on the flat trace; per-Mtoken = co2_g * 1e6 / tokens.
    car = kavier.inference.carbon(INFER).iloc[0]  # default intensity 400 gCO2/kWh
    assert car["total_co2_g"] == pytest.approx(car["total_energy_kwh"] * 400.0)
    assert car["total_co2_kg"] == pytest.approx(car["total_co2_g"] / 1000.0)
    per_m = car["total_co2_g"] * 1_000_000.0 / (128 * 640)
    assert car["carbon_per_mtoken_g"] == pytest.approx(per_m)


def test_inference_carbon_scales_linearly_with_grid_intensity() -> None:
    # Energy does not depend on intensity, so doubling gCO2/kWh doubles emissions.
    base = kavier.inference.carbon({**INFER, "intensity": 400.0}).iloc[0]
    dbl = kavier.inference.carbon({**INFER, "intensity": 800.0}).iloc[0]
    assert dbl["total_co2_g"] == pytest.approx(2.0 * base["total_co2_g"])
    assert dbl["total_energy_kwh"] == pytest.approx(base["total_energy_kwh"])


def test_inference_cost_per_mtoken_from_gpu_hours_and_price() -> None:
    # $/Mtoken = gpu_hours * $/gpu-hour * 1e6 / tokens; gpu_hours = total_s / 3600. Default price 2.5.
    perf = kavier.inference.performance(INFER).iloc[0]
    eff = kavier.inference.efficiency(INFER).iloc[0]
    gpu_hours = perf["total_s"] / 3600.0
    assert eff["gpu_hours"] == pytest.approx(gpu_hours)
    assert eff["financial_per_mtoken"] == pytest.approx(gpu_hours * 2.5 * 1_000_000.0 / (128 * 640))
    # A gpu_hour_price column overrides the 2.5 default: 4x the price -> 4x the cost.
    dearer = kavier.inference.efficiency({**INFER, "gpu_hour_price": 10.0}).iloc[0]
    assert dearer["financial_per_mtoken"] == pytest.approx(4.0 * eff["financial_per_mtoken"])


def test_inference_inputs_dict_dataframe_and_list_are_equivalent() -> None:
    # docs/usage.py section 3: a dict, a 1-row DataFrame and a 1-element list give equal predictions.
    as_dict = kavier.inference.performance(INFER).iloc[0]
    as_df = kavier.inference.performance(pd.DataFrame([INFER])).iloc[0]
    as_list = kavier.inference.performance([INFER]).iloc[0]
    for col in ("p50_ms", "throughput_tok_s", "total_s", "total_tokens"):
        assert as_df[col] == pytest.approx(as_dict[col])
        assert as_list[col] == pytest.approx(as_dict[col])


@pytest.mark.parametrize(
    "bad",
    [
        {**INFER, "num_requests": 0},  # fewer than 1 request
        {**INFER, "input_tokens": -1},  # negative token count
        {**INFER, "output_tokens": -5},
    ],
)
def test_inference_rejects_unsimulatable_workloads(bad: dict) -> None:
    with pytest.raises(ValueError):
        kavier.inference.performance(bad)


# --------------------------------------------------------------------------- #
# Training verbs
# --------------------------------------------------------------------------- #
def test_training_total_tokens_from_epochs_times_dataset() -> None:
    # Job size can be given as epochs x dataset_tokens: 3 * 5_000_000 = 15_000_000.
    perf = kavier.training.performance(TRAIN_LORA).iloc[0]
    assert perf["total_tokens"] == round(3 * 5_000_000) == 15_000_000
    # Or as total_tokens, which takes precedence over epochs and dataset_tokens.
    direct = kavier.training.performance(
        {**TRAIN_LORA, "total_tokens": 50_000_000, "epochs": None, "dataset_tokens": None}
    ).iloc[0]
    assert direct["total_tokens"] == 50_000_000


def test_training_runtime_is_total_tokens_over_throughput() -> None:
    # runtime = total_tokens / tokens_per_second.
    perf = kavier.training.performance(TRAIN_LORA).iloc[0]
    assert perf["train_runtime"] * perf["train_tokens_per_second"] == pytest.approx(perf["total_tokens"])


@pytest.mark.parametrize("gpu_name", sorted(GPU_SPEC_LIBRARY))
def test_training_power_is_within_gpu_idle_max_bounds(gpu_name: str) -> None:
    # P(u) = idle + (max-idle)*(2u - u^r) with r=1 and u clamped to [0,1] -> P in [idle, max].
    gpu = get_gpu(gpu_name)
    perf = kavier.training.performance({**TRAIN_LORA, "gpu": gpu_name}).iloc[0]
    assert gpu.idle_power_w <= perf["gpu_power_watts"] <= gpu.max_power_w


def test_training_aggregate_power_is_per_gpu_times_total_gpus() -> None:
    # aggregate_power_w = per-GPU power x (num_gpus * num_nodes).
    perf = kavier.training.performance(TRAIN_LORA).iloc[0]  # 8 gpus x 1 node = 8
    ener = kavier.training.energy(TRAIN_LORA).iloc[0]
    assert ener["aggregate_power_w"] == pytest.approx(perf["gpu_power_watts"] * 8)
    # 16 GPUs x 2 nodes = 32 GPUs.
    multinode = {**TRAIN_LORA, "num_gpus": 16, "num_nodes": 2}
    p2 = kavier.training.performance(multinode).iloc[0]
    e2 = kavier.training.energy(multinode).iloc[0]
    assert e2["aggregate_power_w"] == pytest.approx(p2["gpu_power_watts"] * 32)


def test_training_cost_per_mtoken_uses_total_gpu_hours() -> None:
    # $/Mtoken = (runtime_h * total_gpus) * price * 1e6 / total_tokens; default price 2.5, 8 GPUs.
    perf = kavier.training.performance(TRAIN_LORA).iloc[0]
    eff = kavier.training.efficiency(TRAIN_LORA).iloc[0]
    gpu_hours = perf["train_runtime"] / 3600.0 * 8
    assert eff["gpu_hours"] == pytest.approx(gpu_hours)
    assert eff["financial_per_mtoken"] == pytest.approx(gpu_hours * 2.5 * 1_000_000.0 / perf["total_tokens"])
