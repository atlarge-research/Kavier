"""``python -m kavier.cli energy`` subprocess contract (de-slop Phase 0 Deliverable B).

The happy-path math of this subcommand (synthetic powerSource/tasks parquet -> hand-derived Wh / gCO2 /
$ per Mtoken, read back through ``--out`` JSON) is already covered end-to-end by
tests/test_integration/test_cli_contract.py (test_energy_efficiency_math_from_hand_derived_inputs and
friends). This module (mirroring how tests/test_integration/test_cli_cluster.py shells out) complements
that rather than repeating it: it pins the human-readable STDOUT table (kavier/cli/energy.py's
``f"{k:>42}: {v:,.6g}"`` formatting, never asserted verbatim elsewhere) and the two error paths
(missing --kavier/--opendc file, missing ``total_tokens`` column) that no existing test exercises.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Same synthetic fixture as test_energy/test_metrics.py, so the two files' oracles agree:
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

    # kavier/cli/energy.py prints f"{k:>42}: {v:,.6g}" for each summary item -- with our round
    # hand-derived values (500, 150, 40) that format collapses to plain integers. Pin the label:value
    # pairing (not the exact column width, which is cosmetic) -- derived the same way as
    # test_metrics.py, not copied from a run.
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
    # energy/carbon are still printed even though price was omitted.
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
    # Neither --kavier nor --opendc existence is argparse-validated; main() raises FileNotFoundError
    # itself (kavier/cli/energy.py), which surfaces as an uncaught-exception nonzero exit -- pinned as
    # current behaviour (no friendly CLI error message here, unlike e.g. the cluster subcommand).
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
    # cli/energy.py checks for 'total_tokens' explicitly before calling efficiency_summary, and raises
    # a ValueError naming the column (rather than a bare pandas KeyError deeper in the summary code).
    power_path, _ = _write_energy_parquet(tmp_path)
    tasks_path = tmp_path / "tasks.parquet"
    pd.DataFrame({"duration": _DURATION_MS}).to_parquet(tasks_path)  # no total_tokens column

    proc = _run(["--kavier", str(tasks_path), "--opendc", str(power_path)])
    assert proc.returncode != 0
    assert "ValueError" in proc.stderr
    assert "total_tokens" in proc.stderr
