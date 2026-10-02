"""YAML wiring of ``kavier training --config``.

A flat ``{arg_name: value}`` mapping is applied as argparse defaults, so explicit flags still override.
Unknown keys and malformed configs are rejected. Run in a subprocess because ``kavier.cli.training.main``
reads ``sys.argv`` and the config path calls ``parser.error`` / ``sys.exit``; this gives the real exit
code and stderr. Code under test: ``kavier.sdk.io.config.apply_config_defaults``, called from
``kavier.cli._shared.apply_config``.
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

# The _BASE_CFG job as long flags, built without the YAML loader.
_EQUIVALENT_FLAGS = [
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


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "kavier.cli", "training", *args],
        capture_output=True,
        text=True,
    )


def _payload(proc: subprocess.CompletedProcess) -> dict:
    brace = proc.stdout.index("{")
    return json.loads(proc.stdout[brace:])


def test_config_yaml_matches_equivalent_flags(tmp_path):
    # Config and long flags feed one engine; a key dropped or changed by apply_config_defaults makes the
    # payloads differ, or the config run fails on a missing required argument.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_BASE_CFG)

    by_config = _run(["--config", str(cfg)])
    by_flags = _run(_EQUIVALENT_FLAGS)
    assert by_config.returncode == 0, by_config.stderr
    assert by_flags.returncode == 0, by_flags.stderr

    payload = _payload(by_config)
    assert payload == _payload(by_flags)
    # echoed inputs match _BASE_CFG
    assert payload["model_name"] == "mistral-7b-v0.1"
    assert payload["gpu_name"] == "NVIDIA-A100-SXM4-80GB"
    assert payload["method"] == "lora"
    assert payload["batch_size"] == 4  # YAML int 4 survives without going through argparse type=int
    assert payload["tokens_per_sample"] == 1024
    assert payload["number_gpus"] == 8
    assert payload["total_tokens"] == 10000000


def test_explicit_flag_overrides_config(tmp_path):
    # Config sets batch_size: 4, the flag 8. A set_defaults value loses to an explicit flag, so the result
    # is 8; 4 means the config was applied after parsing.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_BASE_CFG)

    proc = _run(["--config", str(cfg), "--batch_size", "8"])
    assert proc.returncode == 0, proc.stderr
    payload = _payload(proc)
    assert payload["batch_size"] == 8
    # a field not given on the command line still comes from the config
    assert payload["model_name"] == "mistral-7b-v0.1"


def test_unknown_config_key_errors(tmp_path):
    # A key with no argparse dest is rejected; set_defaults would otherwise accept it.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_BASE_CFG + "not_a_real_arg: 5\n")

    proc = _run(["--config", str(cfg)])
    assert proc.returncode == 2  # argparse parser.error -> exit 2
    assert "unknown config key" in proc.stderr.lower()
    assert "not_a_real_arg" in proc.stderr


def test_empty_config_is_noop(tmp_path):
    # An empty YAML file loads as None and load_config returns {}; the run uses the flags only.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")

    proc = _run(["--config", str(cfg), *_EQUIVALENT_FLAGS])
    assert proc.returncode == 0, proc.stderr
    assert _payload(proc)["batch_size"] == 4


def test_non_mapping_config_rejected(tmp_path):
    # A YAML list is not an {arg_name: value} mapping; load_config raises ValueError naming its type.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("- a\n- b\n")

    proc = _run(["--config", str(cfg)])
    assert proc.returncode != 0
    assert "must be a yaml mapping" in proc.stderr.lower()
    assert "got list" in proc.stderr.lower()
