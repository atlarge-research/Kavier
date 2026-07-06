"""End-to-end tests for the ``kavier carbon`` CLI (kavier.cli.carbon.main).

Oracle strategy: the trace intensity is held constant at 150 gCO2/kWh, so the
energy-weighted average intensity the CLI reports MUST be exactly 150 and the
billed CO2 MUST equal energy * 150 -- both independent of the (complex) training
engine that produces the energy figure. Where an absolute energy number is
needed, the oracle is either hand-derived (powerSource mode) or the SDK layer
(``fragments_from_training`` + ``compute_emissions``) invoked with explicitly
typed params, which cross-checks that ``main`` forwards its parsed flags
correctly and formats the totals it gets back.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from kavier.cli.carbon import main
from kavier.sdk.co2.engine import compute_emissions, load_carbon_trace
from kavier.sdk.co2.fragments import fragments_from_training

INTENSITY = 150.0  # constant gCO2/kWh over the whole synthetic trace

# One canonical from-training workload, shared by the CLI (as argv strings) and
# the SDK oracle (as typed kwargs) so a wiring bug in main() shows up as a mismatch.
TRAIN_PARAMS = {
    "model_name": "mistral-7b-v0.1",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 1024,
    "batch_size": 4,
    "number_gpus": 8,
    "number_nodes": 1,
    "total_tokens": 1_000_000,
    "start_time": "2025-06-01 00:00",
}


def _train_argv(trace: str, **overrides: object) -> list[str]:
    params = {**TRAIN_PARAMS, **overrides}
    argv = ["--from-training", "--carbon_trace", trace]
    for key, val in params.items():
        argv += [f"--{key}", str(val)]
    return argv


def _parse_totals(out: str) -> dict[str, float]:
    def grab(pattern: str) -> float:
        m = re.search(pattern, out)
        assert m is not None, f"missing line for {pattern!r} in:\n{out}"
        return float(m.group(1).replace(",", ""))

    return {
        "energy_kwh": grab(r"Total energy:\s+([\d,.]+) kWh"),
        "co2_g": grab(r"Total CO2:\s+([\d,.]+) g"),
        "avg_intensity": grab(r"Avg intensity used:\s+([\d,.]+) gCO2/kWh"),
    }


@pytest.fixture()
def small_trace(tmp_path):
    ts = pd.date_range("2025-06-01 00:00", periods=2000, freq="30min")
    df = pd.DataFrame({"timestamp": ts, "carbon_intensity": [INTENSITY] * len(ts)})
    p = tmp_path / "carbon.parquet"
    df.to_parquet(p)
    return str(p)


def test_cli_from_training_bills_at_constant_trace_intensity(small_trace, capsys):
    """On a flat 150 gCO2/kWh trace the reported avg intensity is exactly 150 and CO2 = energy * 150."""
    main(_train_argv(small_trace))
    totals = _parse_totals(capsys.readouterr().out)

    # Hand oracle: a constant-intensity trace forces the energy-weighted mean to
    # equal that intensity, whatever the training engine reports for energy.
    assert totals["avg_intensity"] == INTENSITY
    # CO2 (g) must be energy (kWh) * 150; abs tol covers the 2-decimal display rounding of CO2.
    assert totals["co2_g"] == pytest.approx(totals["energy_kwh"] * INTENSITY, abs=1e-2)


def test_cli_from_training_forwards_args_to_sdk(small_trace, capsys):
    """main() forwards its flags unmangled: CLI totals match a direct SDK computation with the same params."""
    main(_train_argv(small_trace))
    totals = _parse_totals(capsys.readouterr().out)

    # Independent oracle: the SDK layer (different module from the CLI) run with
    # explicitly typed params. A mis-forwarded flag (e.g. swapping batch_size and
    # tokens_per_sample) would change the energy and break this cross-check.
    frags = fragments_from_training(
        model_name=TRAIN_PARAMS["model_name"],
        method=TRAIN_PARAMS["method"],
        gpu_model=TRAIN_PARAMS["gpu_model"],
        tokens_per_sample=TRAIN_PARAMS["tokens_per_sample"],
        batch_size=TRAIN_PARAMS["batch_size"],
        number_gpus=TRAIN_PARAMS["number_gpus"],
        number_nodes=TRAIN_PARAMS["number_nodes"],
        total_tokens=TRAIN_PARAMS["total_tokens"],
        start_time=pd.Timestamp(TRAIN_PARAMS["start_time"]),
    )
    expected = compute_emissions(frags, load_carbon_trace(small_trace))

    # Printed energy is rounded to 4 decimals, CO2 to 2 decimals.
    assert totals["energy_kwh"] == pytest.approx(expected.total_energy_kwh, abs=1e-4)
    assert totals["co2_g"] == pytest.approx(expected.total_co2_g, abs=1e-2)
    # Sanity: the training run consumed real, positive energy (guards a zero SDK stub).
    assert expected.total_energy_kwh > 0


def test_cli_powersource_hand_derived_totals(small_trace, tmp_path, capsys):
    """powerSource mode: 3 windows of 3.6e6 Ws each => 3 kWh total, 450 g CO2 at 150 gCO2/kWh."""
    ps = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-06-01 00:00", "2025-06-01 00:30", "2025-06-01 01:00"]),
            "energy_usage": [3.6e6, 3.6e6, 3.6e6],  # watt-seconds
        }
    )
    ps_path = tmp_path / "powerSource.parquet"
    ps.to_parquet(ps_path)

    main(["--powersource", str(ps_path), "--carbon_trace", small_trace])
    totals = _parse_totals(capsys.readouterr().out)

    # Hand oracle: 3.6e6 Ws / 3.6e6 (Ws per kWh) = 1 kWh per window * 3 = 3 kWh;
    # 3 kWh * 150 gCO2/kWh = 450 g.
    assert totals["energy_kwh"] == pytest.approx(3.0, abs=1e-4)
    assert totals["co2_g"] == pytest.approx(450.0, abs=1e-2)
    assert totals["avg_intensity"] == INTENSITY


def test_cli_output_csv_breakdown_tiles_to_total(small_trace, tmp_path, capsys):
    """--output_csv writes a per-window breakdown whose rows sum back to the reported totals."""
    csv_path = tmp_path / "breakdown.csv"
    main(_train_argv(small_trace, output_csv=str(csv_path)))
    totals = _parse_totals(capsys.readouterr().out)

    assert csv_path.exists()
    bd = pd.read_csv(csv_path)
    assert {"window_start", "carbon_intensity", "energy_kwh", "co2_g"} <= set(bd.columns)
    assert len(bd) >= 1
    # Independent oracle: constant trace => every billed window intensity is 150.
    assert (bd["carbon_intensity"] == INTENSITY).all()
    # The breakdown must tile the whole run: per-window energy/CO2 sum to the printed totals.
    assert bd["energy_kwh"].sum() == pytest.approx(totals["energy_kwh"], abs=1e-4)
    assert bd["co2_g"].sum() == pytest.approx(totals["co2_g"], abs=1e-2)


def test_cli_start_before_trace_coverage_exits_2(small_trace, capsys):
    """A start_time outside the trace window is rejected via a non-zero exit, not silently billed."""
    with pytest.raises(SystemExit) as exc:
        # 2020 is years before the trace's 2025 coverage.
        main(_train_argv(small_trace, start_time="2020-01-01 00:00"))
    assert exc.value.code == 2
    err = capsys.readouterr().err.lower()
    assert "coverage" in err or "range" in err


def test_cli_unknown_model_exits_2(small_trace, capsys):
    """An unknown model name is caught (UnknownSpecError) and turned into exit code 2 with a message."""
    with pytest.raises(SystemExit) as exc:
        main(_train_argv(small_trace, model_name="NoSuchModel-999"))
    assert exc.value.code == 2
    # Falsifies removal of the except-UnknownSpecError branch (a bare KeyError would propagate instead).
    assert "unknown model" in capsys.readouterr().err.lower()


def test_cli_from_training_requires_start_time(small_trace, capsys):
    """--from-training without --start_time errors out (exit 2) naming the missing flag."""
    argv = [a for a in _train_argv(small_trace) if a not in ("--start_time", str(TRAIN_PARAMS["start_time"]))]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "--start_time" in capsys.readouterr().err


def test_cli_requires_a_mode(small_trace, capsys):
    """Neither --from-training nor --powersource is an error: exactly one input mode is required."""
    with pytest.raises(SystemExit) as exc:
        main(["--carbon_trace", small_trace])
    assert exc.value.code == 2
    assert "required" in capsys.readouterr().err.lower()
