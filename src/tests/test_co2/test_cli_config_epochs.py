"""``kavier-co2`` additive inputs: ``--epochs``/``--dataset_tokens`` parity and ``--config`` YAML."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from kavier_co2.cli import main

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
    ts = pd.date_range("2025-06-01 00:00", periods=4000, freq="30min")
    df = pd.DataFrame({"timestamp": ts, "carbon_intensity": [150.0] * len(ts)})
    p = tmp_path / "carbon.parquet"
    df.to_parquet(p)
    return str(p)


def _co2_grams(out: str) -> float:
    m = re.search(r"Total CO2:\s+([\d,.]+) g", out)
    assert m, f"no 'Total CO2' line in output:\n{out}"
    return float(m.group(1).replace(",", ""))


def test_epochs_dataset_tokens_match_total_tokens(small_trace, capsys):
    # 2 epochs x 5,000,000 tokens == 10,000,000 total tokens -> identical emissions.
    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "10000000"])
    by_total = _co2_grams(capsys.readouterr().out)

    main(
        ["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--epochs", "2", "--dataset_tokens", "5000000"]
    )
    by_epochs = _co2_grams(capsys.readouterr().out)

    assert by_epochs == pytest.approx(by_total)
    assert by_total > 0


def test_missing_token_source_errors(small_trace, capsys):
    # Neither --total_tokens nor --epochs/--dataset_tokens -> clear required error.
    with pytest.raises(SystemExit):
        main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS])
    err = capsys.readouterr().err
    assert "total_tokens" in err


def test_config_yaml_matches_flags(small_trace, tmp_path, capsys):
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


def test_config_unknown_key_errors(small_trace, tmp_path, capsys):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("model_name: mistral-7b-v0.1\nbogus_key: 1\n")
    with pytest.raises(SystemExit):
        main(["--from-training", "--carbon_trace", small_trace, "--config", str(cfg)])
    err = capsys.readouterr().err
    assert "unknown config key" in err.lower()
    assert "bogus_key" in err
