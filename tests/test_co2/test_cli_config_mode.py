"""``kavier carbon --config``: config values are checked like flags, can fill required flags, and lose to
flags typed on the command line, including the --from-training / --powersource choice."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from kavier.cli.carbon import main

_JOB_YAML = (
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


@pytest.fixture()
def carbon_trace(tmp_path):
    ts = pd.date_range("2025-06-01 00:00", periods=500, freq="30min")
    p = tmp_path / "carbon.parquet"
    pd.DataFrame({"timestamp": ts, "carbon_intensity": [150.0] * len(ts)}).to_parquet(p)
    return str(p)


@pytest.fixture()
def powersource(tmp_path):
    # 1 kWh per 30-min row, so this run bills far more CO2 than the training job.
    ts = pd.date_range("2025-06-01 00:00", periods=4, freq="30min")
    p = tmp_path / "powerSource.parquet"
    pd.DataFrame({"timestamp": ts, "energy_usage": [3.6e6] * len(ts)}).to_parquet(p)
    return str(p)


def _co2_grams(out: str) -> float:
    m = re.search(r"Total CO2:\s+([\d,.]+) g", out)
    assert m, f"no 'Total CO2' line in output:\n{out}"
    return float(m.group(1).replace(",", ""))


def test_from_training_flag_beats_powersource_in_config(tmp_path, carbon_trace, powersource, capsys):
    cfg = tmp_path / "job.yaml"
    cfg.write_text(_JOB_YAML)
    main(["--config", str(cfg), "--carbon_trace", carbon_trace, "--from-training"])
    by_training = _co2_grams(capsys.readouterr().out)

    cfg_with_ps = tmp_path / "job_ps.yaml"
    cfg_with_ps.write_text(_JOB_YAML + f"powersource: {powersource}\n")
    main(["--config", str(cfg_with_ps), "--carbon_trace", carbon_trace, "--from-training"])
    assert _co2_grams(capsys.readouterr().out) == pytest.approx(by_training)


def test_config_can_supply_carbon_trace_and_mode(tmp_path, carbon_trace, capsys):
    cfg = tmp_path / "job.yaml"
    cfg.write_text(_JOB_YAML + f"carbon_trace: {carbon_trace}\nfrom_training: true\n")
    main(["--config", str(cfg)])
    assert _co2_grams(capsys.readouterr().out) > 0


def test_config_method_outside_choices_is_rejected(tmp_path, carbon_trace, capsys):
    cfg = tmp_path / "job.yaml"
    cfg.write_text(_JOB_YAML.replace("method: lora", "method: bogus"))
    with pytest.raises(SystemExit) as ei:
        main(["--config", str(cfg), "--carbon_trace", carbon_trace, "--from-training"])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err and "bogus" in err
