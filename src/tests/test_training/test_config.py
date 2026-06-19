"""``kavier-train --config`` YAML wiring: defaults, override, and unknown-key errors.

Driven via subprocess because ``kavier_training.cli.main`` reads ``sys.argv`` directly.
"""

from __future__ import annotations

import json
import subprocess
import sys

_BASE_CFG = """\
model_name: mistral-7b-v0.1
method: lora
gpu_model: NVIDIA-A100-SXM4-80GB
tokens_per_sample: 1024
batch_size: 4
number_gpus: 8
number_nodes: 1
total_tokens: 10000000
"""


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "kavier_training.cli", *args],
        capture_output=True,
        text=True,
    )


def _payload(proc: subprocess.CompletedProcess) -> dict:
    brace = proc.stdout.index("{")
    return json.loads(proc.stdout[brace:])


def _equivalent_flags() -> list[str]:
    return [
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
        "10000000",
    ]


def test_config_yaml_matches_equivalent_flags(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_BASE_CFG)

    by_config = _run(["--config", str(cfg)])
    by_flags = _run(_equivalent_flags())
    assert by_config.returncode == 0, by_config.stderr
    assert by_flags.returncode == 0, by_flags.stderr
    # Identical JSON result: the config path is exactly the long-flag path.
    assert _payload(by_config) == _payload(by_flags)


def test_explicit_flag_overrides_config(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_BASE_CFG)  # batch_size: 4

    proc = _run(["--config", str(cfg), "--batch_size", "8"])
    assert proc.returncode == 0, proc.stderr
    payload = _payload(proc)
    assert payload["batch_size"] == 8  # CLI flag beats the YAML default


def test_unknown_config_key_errors(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_BASE_CFG + "not_a_real_arg: 5\n")

    proc = _run(["--config", str(cfg)])
    assert proc.returncode != 0
    assert "unknown config key" in proc.stderr.lower()
    assert "not_a_real_arg" in proc.stderr
