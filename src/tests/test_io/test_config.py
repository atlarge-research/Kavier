"""Unit tests for the shared YAML run-config loader (``kavier_io.config``)."""

from __future__ import annotations

import argparse

import pytest

from kavier_io.config import apply_config_defaults, load_config


def test_load_config_reads_mapping(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model_name: mistral-7b-v0.1\nbatch_size: 4\n")
    assert load_config(str(p)) == {"model_name": "mistral-7b-v0.1", "batch_size": 4}


def test_load_config_empty_file_is_empty_mapping(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_config(str(p)) == {}


def test_load_config_rejects_non_mapping(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_config(str(p))


def test_apply_config_defaults_sets_defaults(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--batch_size", type=int)
    p = tmp_path / "cfg.yaml"
    p.write_text("model_name: foo\nbatch_size: 7\n")

    apply_config_defaults(parser, str(p))
    # With nothing on the command line, the YAML values are the parsed defaults.
    args = parser.parse_args([])
    assert args.model_name == "foo"
    assert args.batch_size == 7
    # An explicit flag still wins over the YAML default.
    overridden = parser.parse_args(["--batch_size", "1"])
    assert overridden.batch_size == 1


def test_apply_config_defaults_unknown_key_errors(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name")
    p = tmp_path / "cfg.yaml"
    p.write_text("model_name: foo\nnope: 1\n")
    with pytest.raises(SystemExit):
        apply_config_defaults(parser, str(p))
