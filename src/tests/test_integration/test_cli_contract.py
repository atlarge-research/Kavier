"""CLI contract: each console script runs end-to-end on shipped data, exits 0,
and emits parseable output. This is the only coverage that exercises the full
wiring (arg parsing -> engine -> OpenDC parquet -> efficiency) the way a user does.

kavier-perf -> kavier-energy is the real pipeline: the inference run must write a
tasks.parquet that the energy calculator can consume (the total_tokens contract).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

SRC = Path(__file__).resolve().parent.parent.parent
INFERENCE_EXAMPLE = SRC / "kavier_inference" / "data" / "input" / "input_example.csv"
TRAINING_EXAMPLE = SRC / "kavier_training" / "data" / "input" / "input_example.csv"


def _run(module: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------- #
# kavier-perf
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not INFERENCE_EXAMPLE.exists(), reason="inference example trace missing")
def test_kavier_perf_runs_and_writes_opendc(tmp_path) -> None:
    proc = _run(
        "kavier_inference.cli",
        ["--trace", str(INFERENCE_EXAMPLE), "--output_folder", str(tmp_path), "--llm", "Llama-3-8B", "--gpu", "A10"],
    )
    assert proc.returncode == 0, proc.stderr
    assert "SIMULATION SUMMARY" in (proc.stdout + proc.stderr)

    # Exactly one timestamped run dir, with valid OpenDC parquet that carries tokens.
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    tasks = pq.read_table(run_dirs[0] / "tasks.parquet")
    fragments = pq.read_table(run_dirs[0] / "fragments.parquet")
    assert tasks.num_rows == 6  # 6 requests in the shipped example
    assert "total_tokens" in tasks.column_names
    assert fragments.num_rows >= tasks.num_rows


# --------------------------------------------------------------------------- #
# kavier-train
# --------------------------------------------------------------------------- #
def test_kavier_train_single_config_exits_zero_and_emits_json() -> None:
    proc = _run(
        "kavier_training.cli",
        [
            "--model_name",
            "mistral-7b-v0.1",
            "--method",
            "lora",
            "--gpu_model",
            "NVIDIA-A100-SXM4-80GB",
            "--tokens_per_sample",
            "1024",
            "--batch_size",
            "4",
            "--number_gpus",
            "8",
            "--number_nodes",
            "1",
        ],
    )
    assert proc.returncode == 0, proc.stderr
    # The tail of stdout is a JSON object; it must parse and carry positive tps.
    brace = proc.stdout.index("{")
    payload = json.loads(proc.stdout[brace:])
    assert payload["train_tokens_per_second"] > 0
    assert payload["number_gpus"] == 8


def test_kavier_train_unknown_model_exits_nonzero_with_message() -> None:
    proc = _run(
        "kavier_training.cli",
        [
            "--model_name",
            "not-a-real-model",
            "--method",
            "lora",
            "--gpu_model",
            "NVIDIA-A100-SXM4-80GB",
            "--tokens_per_sample",
            "1024",
            "--batch_size",
            "4",
            "--number_gpus",
            "1",
            "--number_nodes",
            "1",
        ],
    )
    assert proc.returncode != 0
    assert "not-a-real-model" in proc.stderr


@pytest.mark.skipif(not TRAINING_EXAMPLE.exists(), reason="training example csv missing")
def test_kavier_train_csv_batch_exits_zero() -> None:
    proc = _run("kavier_training.cli", ["--input_csv", str(TRAINING_EXAMPLE)])
    assert proc.returncode == 0, proc.stderr
    assert "configurations simulated" in proc.stdout


# --------------------------------------------------------------------------- #
# Full pipeline: kavier-perf -> kavier-energy
# --------------------------------------------------------------------------- #
def _perf_to_tasks_parquet(tmp_path) -> Path:
    """Run kavier-perf on the shipped example, return the tasks.parquet path."""
    perf = _run(
        "kavier_inference.cli",
        ["--trace", str(INFERENCE_EXAMPLE), "--output_folder", str(tmp_path), "--llm", "Llama-3-8B", "--gpu", "A10"],
    )
    assert perf.returncode == 0, perf.stderr
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    return run_dir / "tasks.parquet"


@pytest.mark.skipif(not INFERENCE_EXAMPLE.exists(), reason="inference example trace missing")
def test_perf_then_energy_pipeline_prints_summary(tmp_path) -> None:
    tasks_path = _perf_to_tasks_parquet(tmp_path)

    # Fabricate a minimal OpenDC powerSource.parquet (energy in Joules, carbon in g).
    power = pd.DataFrame({"energy_usage": [3.6e6, 3.6e6], "carbon_emission": [10.0, 5.0]})
    power_path = tmp_path / "powerSource.parquet"
    power.to_parquet(power_path)

    # Feed both to kavier-energy WITHOUT --out (the stdout path) and confirm it
    # exits 0 with the three efficiency metrics printed (all finite, positive).
    energy = _run(
        "kavier_energy.calculator",
        ["--kavier", str(tasks_path), "--opendc", str(power_path), "--price", "2.0"],
    )
    assert energy.returncode == 0, energy.stderr
    out = energy.stdout
    assert "Efficiency summary" in out
    assert "energy_efficiency (Wh/Mtoken)" in out
    assert "carbon_efficiency (gCO2/Mtoken)" in out
    assert "financial_efficiency ($/Mtoken)" in out


@pytest.mark.skipif(not INFERENCE_EXAMPLE.exists(), reason="inference example trace missing")
def test_perf_then_energy_out_json_serializable(tmp_path) -> None:
    tasks_path = _perf_to_tasks_parquet(tmp_path)
    power = pd.DataFrame({"energy_usage": [3.6e6, 3.6e6], "carbon_emission": [10.0, 5.0]})
    power_path = tmp_path / "powerSource.parquet"
    power.to_parquet(power_path)

    out_json = tmp_path / "summary.json"
    energy = _run(
        "kavier_energy.calculator",
        ["--kavier", str(tasks_path), "--opendc", str(power_path), "--price", "2.0", "--out", str(out_json)],
    )
    assert energy.returncode == 0, energy.stderr
    summary = json.loads(out_json.read_text())
    assert summary["energy_efficiency (Wh/Mtoken)"] > 0
    assert summary["total_tokens"] > 0
