"""Tests for ``kavier carbon`` token-source resolution, ``--config`` folding, and the carbon integral.

The CLI's windowed integral in ``compute_emissions`` is compared with a closed-form energy/CO2 value.
Power and runtime come from ``fragments_from_training`` and serve only as inputs to that reference.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from kavier.cli.carbon import main
from kavier.sdk.co2.fragments import fragments_from_training

# gCO2/kWh. On a flat trace min(own, next) is this value, so the per-window integral has a closed form.
_INTENSITY = 150.0

_BASE_ARGS = [
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
    "--start_time",
    "2025-06-01 00:00",
]


@pytest.fixture()
def small_trace(tmp_path):
    # 4000 half-hour windows, ~83 days: covers every runtime in this module.
    ts = pd.date_range("2025-06-01 00:00", periods=4000, freq="30min")
    df = pd.DataFrame({"timestamp": ts, "carbon_intensity": [_INTENSITY] * len(ts)})
    p = tmp_path / "carbon.parquet"
    df.to_parquet(p)
    return str(p)


def _co2_grams(out: str) -> float:
    m = re.search(r"Total CO2:\s+([\d,.]+) g", out)
    assert m, f"no 'Total CO2' line in output:\n{out}"
    return float(m.group(1).replace(",", ""))


def _closed_form_co2_grams(total_tokens: int) -> float:
    """Return the closed-form CO2 in grams for one fragment inside the flat trace.

    Power and runtime come from the training fragment; the integral is closed form. 1 kWh = 3.6e6 W*s.
    """
    frag = fragments_from_training(
        model_name="mistral-7b-v0.1",
        method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=8,
        number_nodes=1,
        total_tokens=total_tokens,
        start_time=pd.Timestamp("2025-06-01 00:00"),
    )[0]
    energy_kwh = frag.power_w * frag.duration_s / 3.6e6
    return energy_kwh * _INTENSITY


def test_from_training_co2_matches_closed_form(small_trace, capsys):
    # A /1000 (Wh<->kWh) slip or an off-by-one in window accumulation moves the printed grams.
    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "10000000"])
    printed = _co2_grams(capsys.readouterr().out)

    expected = _closed_form_co2_grams(10_000_000)
    # Printed value is rounded to 2 decimals.
    assert printed == pytest.approx(expected, abs=0.01)


def test_epochs_dataset_tokens_parity_with_total_tokens(small_trace, capsys):
    # epochs * dataset_tokens = 2 * 5_000_000 = 10_000_000, the same job size as --total_tokens.
    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "10000000"])
    by_total = _co2_grams(capsys.readouterr().out)

    main(
        ["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--epochs", "2", "--dataset_tokens", "5000000"]
    )
    by_epochs = _co2_grams(capsys.readouterr().out)

    assert by_epochs == pytest.approx(by_total)


def test_co2_scales_linearly_with_total_tokens(small_trace, capsys):
    # runtime = total_tokens / tokens_per_second and energy = power * runtime, so CO2 is linear in tokens.
    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "10000000"])
    co2_10m = _co2_grams(capsys.readouterr().out)

    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "5000000"])
    co2_5m = _co2_grams(capsys.readouterr().out)

    assert co2_5m == pytest.approx(co2_10m / 2.0, rel=1e-3)


def test_missing_token_source_errors(small_trace, capsys):
    # No --total_tokens and no --epochs/--dataset_tokens: parser.error names the token flag.
    with pytest.raises(SystemExit):
        main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS])
    err = capsys.readouterr().err
    assert "total_tokens" in err


def test_epochs_without_dataset_tokens_errors(small_trace, capsys):
    # --epochs needs --dataset_tokens. An OR in the guard would accept this and raise an uncaught ValueError.
    with pytest.raises(SystemExit):
        main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--epochs", "2"])
    err = capsys.readouterr().err
    assert "total_tokens" in err and "dataset_tokens" in err


def test_config_yaml_matches_flags(small_trace, tmp_path, capsys):
    # A config file with the same values as the flags gives the same emissions.
    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "10000000"])
    by_flags = _co2_grams(capsys.readouterr().out)

    cfg = tmp_path / "co2.yaml"
    cfg.write_text(
        "model_name: mistral-7b-v0.1\n"
        "method: lora\n"
        "gpu_model: NVIDIA-A100-SXM4-80GB\n"
        "tokens_per_sample: 1024\n"
        "batch_size: 4\n"
        "number_gpus: 8\n"
        "number_nodes: 1\n"
        "total_tokens: 10000000\n"
        'start_time: "2025-06-01 00:00"\n'
    )
    main(["--from-training", "--carbon_trace", small_trace, "--config", str(cfg)])
    by_config = _co2_grams(capsys.readouterr().out)
    assert by_config == pytest.approx(by_flags)


def test_explicit_flag_overrides_config(small_trace, tmp_path, capsys):
    # Config values are defaults; the explicit flag (10M) wins over the config (5M).
    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "10000000"])
    by_flags_10m = _co2_grams(capsys.readouterr().out)

    cfg = tmp_path / "co2.yaml"
    cfg.write_text(
        "model_name: mistral-7b-v0.1\n"
        "method: lora\n"
        "gpu_model: NVIDIA-A100-SXM4-80GB\n"
        "tokens_per_sample: 1024\n"
        "batch_size: 4\n"
        "number_gpus: 8\n"
        "number_nodes: 1\n"
        "total_tokens: 5000000\n"
        'start_time: "2025-06-01 00:00"\n'
    )
    main(["--from-training", "--carbon_trace", small_trace, "--config", str(cfg), "--total_tokens", "10000000"])
    by_override = _co2_grams(capsys.readouterr().out)
    assert by_override == pytest.approx(by_flags_10m)


def test_config_unknown_key_errors(small_trace, tmp_path, capsys):
    # A key that is not a parser dest is rejected by name.
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("model_name: mistral-7b-v0.1\nbogus_key: 1\n")
    with pytest.raises(SystemExit):
        main(["--from-training", "--carbon_trace", small_trace, "--config", str(cfg)])
    err = capsys.readouterr().err
    assert "unknown config key" in err.lower()
    assert "bogus_key" in err
