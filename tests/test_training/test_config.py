"""``kavier training --config`` YAML wiring: a flat ``{arg_name: value}`` mapping is folded in as argparse
defaults (so explicit flags still override), unknown keys error, and malformed configs are rejected.

Driven via subprocess because ``kavier.cli.training.main`` reads ``sys.argv`` and the config path calls
``parser.error`` / ``sys.exit`` — a subprocess gives us the real exit code and stderr contract. The source
under test is ``kavier.sdk.io.config.apply_config_defaults`` glued in by ``kavier.cli._shared.apply_config``.
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

# The same job expressed as long flags — an independent construction of _BASE_CFG that never touches the
# YAML loader, so it pins the values apply_config_defaults is supposed to inject.
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
    # The YAML path and the long-flag path feed one engine; if apply_config_defaults dropped or mangled any
    # key the two payloads diverge (or the config run errors on a missing required arg). Cross-check + pin.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_BASE_CFG)

    by_config = _run(["--config", str(cfg)])
    by_flags = _run(_EQUIVALENT_FLAGS)
    assert by_config.returncode == 0, by_config.stderr
    assert by_flags.returncode == 0, by_flags.stderr

    payload = _payload(by_config)
    assert payload == _payload(by_flags)  # config path == long-flag path (cross-method check)
    # Independent oracle: the echoed inputs must be exactly what _BASE_CFG declared, not argparse defaults.
    assert payload["model_name"] == "mistral-7b-v0.1"
    assert payload["gpu_name"] == "NVIDIA-A100-SXM4-80GB"
    assert payload["method"] == "lora"
    assert payload["batch_size"] == 4  # YAML int 4 survives without going through argparse type=int
    assert payload["tokens_per_sample"] == 1024
    assert payload["number_gpus"] == 8
    assert payload["total_tokens"] == 10000000


def test_explicit_flag_overrides_config(tmp_path):
    # Config supplies batch_size: 4; the CLI flag says 8. argparse's contract is "set_defaults value loses to
    # an explicit flag", so the result must be 8 — and 4 iff the fold happened AFTER parsing (the bug).
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_BASE_CFG)

    proc = _run(["--config", str(cfg), "--batch_size", "8"])
    assert proc.returncode == 0, proc.stderr
    payload = _payload(proc)
    assert payload["batch_size"] == 8  # CLI flag beats the YAML default
    # ...while a field NOT given on the CLI still comes from the config (override is per-key, not all-or-nothing).
    assert payload["model_name"] == "mistral-7b-v0.1"


def test_unknown_config_key_errors(tmp_path):
    # A key that maps to no argparse dest must be rejected (else set_defaults would silently attach garbage).
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_BASE_CFG + "not_a_real_arg: 5\n")

    proc = _run(["--config", str(cfg)])
    assert proc.returncode == 2  # argparse parser.error -> exit 2
    assert "unknown config key" in proc.stderr.lower()
    assert "not_a_real_arg" in proc.stderr  # the offending key is named


def test_empty_config_is_noop(tmp_path):
    # An empty YAML file parses to None -> load_config returns {}; the run must fall back entirely to the flags
    # (rc 0 with the flag values), not crash on `None is not a mapping`.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")

    proc = _run(["--config", str(cfg), *_EQUIVALENT_FLAGS])
    assert proc.returncode == 0, proc.stderr
    assert _payload(proc)["batch_size"] == 4  # came from the flags, config contributed nothing


def test_non_mapping_config_rejected(tmp_path):
    # A YAML list is not an {arg_name: value} mapping; load_config raises ValueError naming the actual type.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("- a\n- b\n")

    proc = _run(["--config", str(cfg)])
    assert proc.returncode != 0
    assert "must be a yaml mapping" in proc.stderr.lower()
    assert "got list" in proc.stderr.lower()  # the rejected type is reported
