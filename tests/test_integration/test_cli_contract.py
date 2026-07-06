"""CLI contract: each ``kavier`` subcommand runs end-to-end on shipped data and honours its wiring
contract (args -> engine -> OpenDC parquet -> efficiency/carbon). This is the only coverage of the full
wiring, so the oracles here are hand-derived from the *input* data (CSV token counts, synthetic
powerSource / carbon traces) rather than snapshots of engine output.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

# Resolve the shipped example data via importlib.resources (works under both an editable install and
# an installed wheel), coerced to a filesystem Path for .exists() and subprocess arguments.
INFERENCE_EXAMPLE = Path(str(files("kavier.sdk.inference").joinpath("data", "input", "input_example.csv")))
TRAINING_EXAMPLE = Path(str(files("kavier.sdk.training").joinpath("data", "input", "input_example.csv")))


def _run(command: str, args: list[str]) -> subprocess.CompletedProcess:
    """Invoke a ``kavier`` subcommand (inference/training/energy/carbon) via ``python -m kavier.cli``."""
    return subprocess.run(
        [sys.executable, "-m", "kavier.cli", command, *args],
        capture_output=True,
        text=True,
    )


def _run_inference(tmp_path: Path) -> Path:
    """Run ``kavier inference`` on the shipped trace into ``tmp_path``; return the single run directory."""
    proc = _run(
        "inference",
        ["--trace", str(INFERENCE_EXAMPLE), "--output_folder", str(tmp_path), "--llm", "Llama-3-8B", "--gpu", "A10"],
    )
    assert proc.returncode == 0, proc.stderr
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1  # one timestamped run directory per invocation
    return run_dirs[0]


# ----------------------------------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not INFERENCE_EXAMPLE.exists(), reason="inference example trace missing")
def test_inference_carries_csv_tokens_into_tasks_parquet(tmp_path) -> None:
    # Oracle from the *input* CSV, independent of the engine: the OpenDC ``tasks`` table has one row per
    # request, and each task's total_tokens is the request's prompt+completion length. The shipped trace
    # has 6 requests summing to 84 tokens (14+17+15+14+12+12); fragments must tile at least one per task.
    csv = pd.read_csv(INFERENCE_EXAMPLE)
    expected_rows = len(csv)  # 6
    expected_total_tokens = int((csv["num_input_tokens"] + csv["num_output_tokens"]).sum())  # 84
    assert (expected_rows, expected_total_tokens) == (6, 84)  # guards the fixture against silent edits

    run_dir = _run_inference(tmp_path)
    tasks = pq.read_table(run_dir / "tasks.parquet").to_pandas()
    fragments = pq.read_table(run_dir / "fragments.parquet")

    assert len(tasks) == expected_rows
    assert int(tasks["total_tokens"].sum()) == expected_total_tokens
    assert fragments.num_rows >= len(tasks)  # invariant: >=1 fragment per task


@pytest.mark.skipif(not INFERENCE_EXAMPLE.exists(), reason="inference example trace missing")
def test_inference_prints_summary_banner(tmp_path) -> None:
    # Contract: the run reaches the summary printer (not a silent/partial exit).
    proc = _run(
        "inference",
        ["--trace", str(INFERENCE_EXAMPLE), "--output_folder", str(tmp_path), "--llm", "Llama-3-8B", "--gpu", "A10"],
    )
    assert proc.returncode == 0, proc.stderr
    assert "SIMULATION SUMMARY" in (proc.stdout + proc.stderr)


# ----------------------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------------------


def test_training_json_fields_are_internally_consistent() -> None:
    # Use 4 GPUs x 2 nodes so the "total gpus vs per-node gpus" distinction is observable: the reported
    # ``number_gpus`` must be the TOTAL (8), catching a bug that echoes the per-node count (4).
    proc = _run(
        "training",
        [
            "--model_name", "mistral-7b-v0.1",
            "--method", "lora",
            "--gpu_model", "NVIDIA-A100-SXM4-80GB",
            "--tokens_per_sample", "1024",
            "--batch_size", "4",
            "--number_gpus", "4",
            "--number_nodes", "2",
            "--total_tokens", "100000000",
        ],
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.index("{") :])

    assert payload["number_gpus"] == 4 * 2  # total gpus, not per-node
    tps = payload["train_tokens_per_second"]
    assert tps > 0
    # Defining relationships between the reported fields (independent cross-checks; a mutation in any one
    # derived field breaks its identity):
    # per-gpu throughput * total gpus == aggregate throughput.
    assert payload["train_tokens_per_gpu_per_second"] * 8 == pytest.approx(tps)
    # samples/s * tokens_per_sample == tokens/s.
    assert payload["train_samples_per_second"] * 1024 == pytest.approx(tps)
    # runtime = total_tokens / throughput.
    assert payload["train_runtime"] * tps == pytest.approx(100_000_000)
    # input echoes flow through unchanged.
    assert payload["model_name"] == "mistral-7b-v0.1"
    assert payload["method"] == "lora"


def test_training_unknown_model_exits_nonzero_with_message() -> None:
    proc = _run(
        "training",
        [
            "--model_name", "not-a-real-model",
            "--method", "lora",
            "--gpu_model", "NVIDIA-A100-SXM4-80GB",
            "--tokens_per_sample", "1024",
            "--batch_size", "4",
            "--number_gpus", "1",
            "--number_nodes", "1",
        ],
    )  # fmt: skip
    assert proc.returncode != 0
    assert "not-a-real-model" in proc.stderr


@pytest.mark.skipif(not TRAINING_EXAMPLE.exists(), reason="training example csv missing")
def test_training_csv_batch_reports_row_count() -> None:
    # Oracle from the input CSV: the batch runner simulates exactly one config per data row and reports
    # that count. Reading the row count here (not a magic literal) falsifies a runner that skips/dupes rows.
    n_rows = len(pd.read_csv(TRAINING_EXAMPLE))
    proc = _run("training", ["--input_csv", str(TRAINING_EXAMPLE)])
    assert proc.returncode == 0, proc.stderr
    assert f"{n_rows} configurations simulated." in proc.stdout


# ----------------------------------------------------------------------------------------------------
# energy
# ----------------------------------------------------------------------------------------------------


def _write_synthetic_tasks(path: Path, total_tokens: list[int], duration_ms: list[float]) -> None:
    pd.DataFrame({"total_tokens": total_tokens, "duration": duration_ms}).to_parquet(path)


def _write_powersource(path: Path, energy_j: list[float], carbon_g: list[float]) -> None:
    pd.DataFrame({"energy_usage": energy_j, "carbon_emission": carbon_g}).to_parquet(path)


def test_energy_efficiency_math_from_hand_derived_inputs(tmp_path) -> None:
    # Fully hand-derived oracle, decoupled from the inference engine by feeding a synthetic tasks table.
    #   total_tokens = 500000 + 500000 = 1_000_000 (1 Mtoken)  -> the "/Mtoken" scaling is x1
    #   energy       = (3.6e6 + 3.6e6) J / 3600 (J per Wh) = 2000 Wh   (the old /1000 bug -> 7200 Wh)
    #   carbon       = 10 + 5 = 15 g          (OpenDC carbon is already grams, no conversion)
    #   gpu-hours    = (1.8e6 + 1.8e6) ms / 3.6e6 = 1.0 h ; cost = 1.0 h * $2 = $2 per Mtoken
    tasks_path = tmp_path / "tasks.parquet"
    ps_path = tmp_path / "powerSource.parquet"
    _write_synthetic_tasks(tasks_path, total_tokens=[500_000, 500_000], duration_ms=[1.8e6, 1.8e6])
    _write_powersource(ps_path, energy_j=[3.6e6, 3.6e6], carbon_g=[10.0, 5.0])

    out_json = tmp_path / "summary.json"
    proc = _run(
        "energy",
        ["--kavier", str(tasks_path), "--opendc", str(ps_path), "--price", "2.0", "--out", str(out_json)],
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(out_json.read_text())

    assert summary["total_tokens"] == 1_000_000
    assert summary["energy_efficiency (Wh/Mtoken)"] == pytest.approx(2000.0)
    assert summary["energy_efficiency (Wh/Mtoken)"] != pytest.approx(7200.0)  # old /1000-instead-of-/3600 bug
    assert summary["carbon_efficiency (gCO2/Mtoken)"] == pytest.approx(15.0)
    assert summary["financial_efficiency ($/Mtoken)"] == pytest.approx(2.0)


def test_energy_financial_is_none_without_price(tmp_path) -> None:
    # Contract: cost is reported ONLY when --price is set; otherwise it is null in JSON and "N/A" on stdout.
    tasks_path = tmp_path / "tasks.parquet"
    ps_path = tmp_path / "powerSource.parquet"
    _write_synthetic_tasks(tasks_path, total_tokens=[1_000_000], duration_ms=[3.6e6])
    _write_powersource(ps_path, energy_j=[3.6e6], carbon_g=[15.0])

    out_json = tmp_path / "summary.json"
    proc = _run("energy", ["--kavier", str(tasks_path), "--opendc", str(ps_path), "--out", str(out_json)])
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(out_json.read_text())

    assert summary["financial_efficiency ($/Mtoken)"] is None
    assert "N/A" in proc.stdout
    # The other metrics are still computed: 3.6e6 J / 3600 = 1000 Wh over 1 Mtoken.
    assert summary["energy_efficiency (Wh/Mtoken)"] == pytest.approx(1000.0)


@pytest.mark.skipif(not INFERENCE_EXAMPLE.exists(), reason="inference example trace missing")
def test_inference_to_energy_pipeline_preserves_token_total(tmp_path) -> None:
    # End-to-end wiring: inference's tasks.parquet feeds energy. The token total that energy sums must be
    # the CSV's 84 tokens (the pipeline neither drops nor invents tokens). The energy/carbon efficiency
    # RATIO is token-independent, so it pins the unit math (2000 Wh vs 15 g) without depending on the
    # engine's absolute output.
    run_dir = _run_inference(tmp_path)
    ps_path = tmp_path / "powerSource.parquet"
    _write_powersource(ps_path, energy_j=[3.6e6, 3.6e6], carbon_g=[10.0, 5.0])  # 2000 Wh, 15 g

    out_json = tmp_path / "summary.json"
    proc = _run(
        "energy",
        ["--kavier", str(run_dir / "tasks.parquet"), "--opendc", str(ps_path), "--out", str(out_json)],
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(out_json.read_text())

    assert summary["total_tokens"] == 84  # sum of prompt+completion over the 6 shipped requests
    ratio = summary["energy_efficiency (Wh/Mtoken)"] / summary["carbon_efficiency (gCO2/Mtoken)"]
    assert ratio == pytest.approx(2000.0 / 15.0)  # == energy_Wh / carbon_g, independent of token count


# ----------------------------------------------------------------------------------------------------
# carbon
# ----------------------------------------------------------------------------------------------------


def test_carbon_from_training_bills_at_trace_intensity(tmp_path) -> None:
    # A constant-intensity trace makes the emission arithmetic checkable without the training engine's
    # absolute energy: the energy-weighted average intensity of a flat 400 gCO2/kWh trace is EXACTLY 400,
    # and total CO2 must equal (total energy kWh) * 400. Windows are 30 min; a full day of coverage from
    # the 00:00 start spans the sub-hour training job.
    trace = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-06-01 00:00", periods=48, freq="30min"),
            "carbon_intensity": 400.0,
        }
    )
    trace_path = tmp_path / "carbon_trace.parquet"
    trace.to_parquet(trace_path)

    proc = _run(
        "carbon",
        [
            "--from-training",
            "--carbon_trace", str(trace_path),
            "--model_name", "mistral-7b-v0.1",
            "--method", "lora",
            "--gpu_model", "NVIDIA-A100-SXM4-80GB",
            "--tokens_per_sample", "1024",
            "--batch_size", "4",
            "--number_gpus", "8",
            "--number_nodes", "1",
            "--total_tokens", "100000000",
            "--start_time", "2025-06-01 00:00",
        ],
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr

    energy_kwh = float(re.search(r"Total energy:\s+([\d.,]+) kWh", proc.stdout).group(1).replace(",", ""))
    co2_g = float(re.search(r"Total CO2:\s+([\d.,]+) g", proc.stdout).group(1).replace(",", ""))
    avg = float(re.search(r"Avg intensity used:\s+([\d.,]+) gCO2/kWh", proc.stdout).group(1).replace(",", ""))

    assert avg == pytest.approx(400.0, abs=0.01)  # energy-weighted mean of a constant trace
    assert co2_g == pytest.approx(energy_kwh * 400.0, rel=1e-3)  # CO2 = energy * intensity (unit check)


# ----------------------------------------------------------------------------------------------------
# root parser / module entry points
# ----------------------------------------------------------------------------------------------------


def test_module_version_matches_package() -> None:
    from kavier import __version__

    proc = subprocess.run([sys.executable, "-m", "kavier", "--version"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert __version__ in proc.stdout  # single-sourced version string, not a hardcoded literal


def test_bare_invocation_exits_two_git_style() -> None:
    # Contract (main.py): a bare `kavier` prints help and exits 2, git-style.
    proc = subprocess.run([sys.executable, "-m", "kavier.cli"], capture_output=True, text=True)
    assert proc.returncode == 2


def test_invalid_subcommand_errors() -> None:
    proc = _run("boguscmd", [])
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr
    assert "boguscmd" in proc.stderr


def test_ui_non_tty_exits_clean_on_eof() -> None:
    # The REPL is POSIX-only; with a non-TTY stdin it must detect EOF and exit 0 rather than crash.
    proc = subprocess.run([sys.executable, "-m", "kavier.ui"], input="", capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
