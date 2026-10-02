"""Subprocess tests for ``python -m kavier.cli energy``: stdout table and error paths.

The ``--out`` JSON values are tested in tests/test_integration/test_cli_contract.py. This module
pins the stdout table (``f"{k:>42}: {v:,.6g}"`` in kavier/cli/energy.py), a missing --kavier or
--opendc file, and a missing ``total_tokens`` column.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Same fixture as test_energy/test_metrics.py.
# energy: 180_000 J / 3600 = 50 Wh; carbon: 15 g; gpu-hours: 3_600_000 ms = 1 h; tokens: 100_000.
_ENERGY_J = [72_000.0, 90_000.0, 18_000.0]
_CARBON_G = [7.0, 5.0, 3.0]
_DURATION_MS = [1_800_000, 1_800_000]
_TOTAL_TOKENS = [60_000, 40_000]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "kavier.cli", "energy", *args], capture_output=True, text=True)


def _write_energy_parquet(tmp_path: Path) -> tuple[Path, Path]:
    power_path = tmp_path / "powerSource.parquet"
    tasks_path = tmp_path / "tasks.parquet"
    pd.DataFrame({"energy_usage": _ENERGY_J, "carbon_emission": _CARBON_G}).to_parquet(power_path)
    pd.DataFrame({"duration": _DURATION_MS, "total_tokens": _TOTAL_TOKENS}).to_parquet(tasks_path)
    return power_path, tasks_path


def test_cli_prints_hand_derived_values_in_the_fixed_width_table(tmp_path: Path) -> None:
    power_path, tasks_path = _write_energy_parquet(tmp_path)
    proc = _run(["--kavier", str(tasks_path), "--opendc", str(power_path), "--price", "4.0"])
    assert proc.returncode == 0, proc.stderr

    # f"{k:>42}: {v:,.6g}" prints 500, 150, 40 as integers. Only the label: value pairing is pinned.
    assert "----------  Efficiency summary  ----------" in proc.stdout
    assert "energy_efficiency (Wh/Mtoken): 500" in proc.stdout
    assert "carbon_efficiency (gCO2/Mtoken): 150" in proc.stdout
    assert "financial_efficiency ($/Mtoken): 40" in proc.stdout
    assert "total_tokens: 100,000" in proc.stdout


def test_cli_prints_na_hint_when_price_omitted(tmp_path: Path) -> None:
    power_path, tasks_path = _write_energy_parquet(tmp_path)
    proc = _run(["--kavier", str(tasks_path), "--opendc", str(power_path)])
    assert proc.returncode == 0, proc.stderr
    assert "financial_efficiency ($/Mtoken): N/A  (pass --price to compute)" in proc.stdout
    # energy and carbon are printed without a price.
    assert "energy_efficiency (Wh/Mtoken): 500" in proc.stdout


def test_cli_out_flag_writes_json_and_prints_save_path(tmp_path: Path) -> None:
    power_path, tasks_path = _write_energy_parquet(tmp_path)
    out_json = tmp_path / "summary.json"
    proc = _run(["--kavier", str(tasks_path), "--opendc", str(power_path), "--out", str(out_json)])
    assert proc.returncode == 0, proc.stderr
    assert out_json.exists()
    assert "Saved" in proc.stdout and str(out_json) in proc.stdout


@pytest.mark.parametrize("missing_flag", ["--kavier", "--opendc"])
def test_cli_missing_input_file_exits_nonzero(tmp_path: Path, missing_flag: str) -> None:
    # argparse does not check the paths; main() raises an uncaught FileNotFoundError.
    # Pins current behaviour; the cluster subcommand prints a friendlier message.
    power_path, tasks_path = _write_energy_parquet(tmp_path)
    args = {
        "--kavier": str(tasks_path),
        "--opendc": str(power_path),
    }
    args[missing_flag] = str(tmp_path / "does-not-exist.parquet")
    proc = _run(["--kavier", args["--kavier"], "--opendc", args["--opendc"]])
    assert proc.returncode != 0
    assert "FileNotFoundError" in proc.stderr
    assert "does-not-exist.parquet" in proc.stderr


def test_cli_missing_total_tokens_column_raises_friendly_value_error(tmp_path: Path) -> None:
    # cli/energy.py checks for 'total_tokens' before efficiency_summary and raises a ValueError naming it.
    power_path, _ = _write_energy_parquet(tmp_path)
    tasks_path = tmp_path / "tasks.parquet"
    pd.DataFrame({"duration": _DURATION_MS}).to_parquet(tasks_path)

    proc = _run(["--kavier", str(tasks_path), "--opendc", str(power_path)])
    assert proc.returncode != 0
    assert "ValueError" in proc.stderr
    assert "total_tokens" in proc.stderr
