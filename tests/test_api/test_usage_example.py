"""``docs/usage.py`` is the public-API spec: ``import kavier`` must expose ``inference``/``training``
verbs that run end-to-end AND return the documented columns with the documented arithmetic.

The integration test proves the script runs; the direct-API tests re-derive each predicted column
from an INDEPENDENT oracle (hand arithmetic, physics, a library constant, an invariant, or a scaling
law) so a wrong formula goes red instead of being snapshotted green.
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

# The two workloads documented in docs/usage.py, reused as fixtures for the direct-API oracles.
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
# Integration: the documented example is the API spec and must run under `import kavier`.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not USAGE.exists(), reason="docs/usage.py missing")
def test_usage_example_script_runs_and_emits_both_tables() -> None:
    # Contract: exit 0 (falsifies if usage.py raises) + it prints BOTH the inference table
    # (throughput_tok_s) and the training table (gpu_power_watts) — proving both sections executed.
    # pandas elides middle columns with "...", so assert on edge columns that actually print:
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
    # Single-stream contract: n IDENTICAL requests => identical latencies => p50 == p95 == p99.
    # Falsifies if any per-request variation (queueing/contention/jitter) crept in.
    perf = kavier.inference.performance(INFER).iloc[0]
    assert perf["p50_ms"] == pytest.approx(perf["p95_ms"])


def test_inference_throughput_is_invariant_under_request_count() -> None:
    # No batching/contention: total_s and total_tokens both scale linearly with num_requests,
    # so throughput_tok_s is independent of it. Falsifies if requests interacted (shared/contended).
    one = kavier.inference.performance({**INFER, "num_requests": 1}).iloc[0]
    many = kavier.inference.performance({**INFER, "num_requests": 256}).iloc[0]
    assert many["throughput_tok_s"] == pytest.approx(one["throughput_tok_s"], rel=1e-9)
    # ...and total_s is exactly linear: 256 identical requests take 256x one request.
    assert many["total_s"] == pytest.approx(256 * one["total_s"], rel=1e-9)


def test_inference_energy_bills_gpu_tdp_over_busy_time() -> None:
    # Inference bills GPU max power (TDP). Physics: Wh = W * s / 3600. A10 TDP = 150 W (library).
    perf = kavier.inference.performance(INFER).iloc[0]
    ener = kavier.inference.energy(INFER).iloc[0]
    tdp_w = get_gpu("A10").max_power_w
    assert tdp_w == 150  # guards the oracle constant against a library edit
    expected_wh = tdp_w * perf["total_s"] / 3600.0
    assert ener["energy_wh"] == pytest.approx(expected_wh)
    # kWh is Wh/1000 (guards the /1000 unit bug).
    assert ener["energy_kwh"] == pytest.approx(ener["energy_wh"] / 1000.0)


def test_inference_carbon_from_energy_and_intensity() -> None:
    # gCO2 = energy(kWh) * intensity(gCO2/kWh) on the flat trace; per-Mtoken = co2_g * 1e6 / tokens.
    car = kavier.inference.carbon(INFER).iloc[0]  # default intensity 400 gCO2/kWh
    assert car["total_co2_g"] == pytest.approx(car["total_energy_kwh"] * 400.0)
    assert car["total_co2_kg"] == pytest.approx(car["total_co2_g"] / 1000.0)
    per_m = car["total_co2_g"] * 1_000_000.0 / (128 * 640)
    assert car["carbon_per_mtoken_g"] == pytest.approx(per_m)


def test_inference_carbon_scales_linearly_with_grid_intensity() -> None:
    # Energy is intensity-independent, so doubling gCO2/kWh doubles emissions exactly.
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
    # A per-row price column overrides the 2.5 default: 4x the price => 4x the cost.
    dearer = kavier.inference.efficiency({**INFER, "gpu_hour_price": 10.0}).iloc[0]
    assert dearer["financial_per_mtoken"] == pytest.approx(4.0 * eff["financial_per_mtoken"])


def test_inference_inputs_dict_dataframe_and_list_are_equivalent() -> None:
    # docs/usage.py section 3: a single dict, a 1-row DataFrame, and a 1-element list must predict
    # identically. Falsifies if _normalise diverged across the three input shapes.
    as_dict = kavier.inference.performance(INFER).iloc[0]
    as_df = kavier.inference.performance(pd.DataFrame([INFER])).iloc[0]
    as_list = kavier.inference.performance([INFER]).iloc[0]
    for col in ("p50_ms", "throughput_tok_s", "total_s", "total_tokens"):
        assert as_df[col] == pytest.approx(as_dict[col])
        assert as_list[col] == pytest.approx(as_dict[col])


@pytest.mark.parametrize(
    "bad",
    [
        {**INFER, "num_requests": 0},  # < 1 request is unsimulatable
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
    # Or as an explicit total_tokens (which wins over epochs/dataset if both present).
    direct = kavier.training.performance(
        {**TRAIN_LORA, "total_tokens": 50_000_000, "epochs": None, "dataset_tokens": None}
    ).iloc[0]
    assert direct["total_tokens"] == 50_000_000


def test_training_runtime_is_total_tokens_over_throughput() -> None:
    # train_runtime * train_tokens_per_second must reconstruct total_tokens (definition of runtime).
    perf = kavier.training.performance(TRAIN_LORA).iloc[0]
    assert perf["train_runtime"] * perf["train_tokens_per_second"] == pytest.approx(perf["total_tokens"])


@pytest.mark.parametrize("gpu_name", sorted(GPU_SPEC_LIBRARY))
def test_training_power_is_within_gpu_idle_max_bounds(gpu_name: str) -> None:
    # P(u) = idle + (max-idle)*(2u - u^r) with r=1 and u clamped to [0,1] => P in [idle, max].
    # Holds for every GPU in the catalogue. Falsifies if power escaped its physical envelope.
    gpu = get_gpu(gpu_name)
    perf = kavier.training.performance({**TRAIN_LORA, "gpu": gpu_name}).iloc[0]
    assert gpu.idle_power_w <= perf["gpu_power_watts"] <= gpu.max_power_w


def test_training_aggregate_power_is_per_gpu_times_total_gpus() -> None:
    # aggregate_power_w bills every GPU: per-GPU power x (num_gpus * num_nodes).
    perf = kavier.training.performance(TRAIN_LORA).iloc[0]  # 8 gpus x 1 node = 8
    ener = kavier.training.energy(TRAIN_LORA).iloc[0]
    assert ener["aggregate_power_w"] == pytest.approx(perf["gpu_power_watts"] * 8)
    # Multi-node: 16 gpus x 2 nodes = 32 total GPUs (not just 16).
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
