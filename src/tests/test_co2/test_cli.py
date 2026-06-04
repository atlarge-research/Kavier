"""End-to-end CLI tests using a small synthetic carbon trace parquet."""

from __future__ import annotations

import pandas as pd
import pytest

from kavier_co2.cli import main


@pytest.fixture()
def small_trace(tmp_path):
    # 2025 covering many windows so a real training runtime fits inside.
    ts = pd.date_range("2025-06-01 00:00", periods=2000, freq="30min")
    df = pd.DataFrame({"timestamp": ts, "carbon_intensity": [150.0] * len(ts)})
    p = tmp_path / "carbon.parquet"
    df.to_parquet(p)
    return str(p)


def test_cli_from_training_prints_totals(small_trace, capsys):
    main(
        [
            "--from-training",
            "--carbon_trace",
            small_trace,
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
            "--total_tokens",
            "1000000",
            "--start_time",
            "2025-06-01 00:00",
        ]
    )
    out = capsys.readouterr().out
    assert "Total energy" in out
    assert "kWh" in out
    assert "Total CO2" in out
    assert "kg" in out
    assert "150" in out  # average intensity is the constant 150


def test_cli_output_csv(small_trace, tmp_path, capsys):
    csv_path = tmp_path / "breakdown.csv"
    main(
        [
            "--from-training",
            "--carbon_trace",
            small_trace,
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
            "--total_tokens",
            "1000000",
            "--start_time",
            "2025-06-01 00:00",
            "--output_csv",
            str(csv_path),
        ]
    )
    assert csv_path.exists()
    bd = pd.read_csv(csv_path)
    assert {"window_start", "carbon_intensity", "energy_kwh", "co2_g"} <= set(bd.columns)
    assert len(bd) >= 1


def test_cli_out_of_range_start_errors(small_trace, capsys):
    with pytest.raises(SystemExit):
        main(
            [
                "--from-training",
                "--carbon_trace",
                small_trace,
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
                "--total_tokens",
                "1000000",
                "--start_time",
                "2020-01-01 00:00",  # before trace coverage
            ]
        )
    err = capsys.readouterr().err
    assert "coverage" in err.lower() or "range" in err.lower()


def test_cli_powersource_mode(small_trace, tmp_path, capsys):
    ps = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-06-01 00:00", "2025-06-01 00:30", "2025-06-01 01:00"]),
            "energy_usage": [3.6e6, 3.6e6, 3.6e6],  # 1 kWh each window
        }
    )
    ps_path = tmp_path / "powerSource.parquet"
    ps.to_parquet(ps_path)
    main(["--powersource", str(ps_path), "--carbon_trace", small_trace])
    out = capsys.readouterr().out
    assert "Total CO2" in out
