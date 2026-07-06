"""Unit tests for the shared YAML run-config loader (``kavier.sdk.io.config``)."""

from __future__ import annotations

import argparse

import pytest

from kavier.sdk.io.config import apply_config_defaults, load_config


def test_load_config_parses_yaml_scalar_types(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model_name: mistral-7b-v0.1\nbatch_size: 4\n")
    result = load_config(str(p))
    # Oracle: YAML scalar rules — an unquoted `4` decodes to int, an unquoted
    # word stays str. So the dict equality only holds if the loader returns the
    # YAML-decoded values (int 4, not the string "4").
    assert result == {"model_name": "mistral-7b-v0.1", "batch_size": 4}
    assert isinstance(result["batch_size"], int)  # falsifies a "return raw text" impl


def test_load_config_empty_file_is_empty_mapping(tmp_path):
    # yaml.safe_load("") is None; the loader's spec normalizes that to {}, not None.
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_config(str(p)) == {}


@pytest.mark.parametrize(
    ("body", "typename"),
    [("- a\n- b\n", "list"), ("42\n", "int"), ("just-a-string\n", "str")],
)
def test_load_config_rejects_non_mapping(tmp_path, body, typename):
    # Contract: only a top-level YAML mapping is a valid config. A sequence or a
    # bare scalar must raise ValueError naming the offending decoded type.
    p = tmp_path / "bad.yaml"
    p.write_text(body)
    with pytest.raises(ValueError) as ei:
        load_config(str(p))
    msg = str(ei.value)
    assert "must be a YAML mapping" in msg
    assert typename in msg  # the offending type is reported, not swallowed


def test_apply_config_defaults_yaml_acts_as_default_not_force(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--batch_size", type=int)
    p = tmp_path / "cfg.yaml"
    p.write_text("model_name: foo\nbatch_size: 7\n")

    apply_config_defaults(parser, str(p))
    # With no CLI args the YAML values surface as the parsed defaults.
    assert parser.parse_args([]).__dict__ == {"model_name": "foo", "batch_size": 7}
    # An explicit flag beats the YAML default — proves set_defaults, not a forced
    # override that would ignore the command line.
    assert parser.parse_args(["--batch_size", "1"]).batch_size == 1


def test_apply_config_defaults_unknown_key_aborts(tmp_path, capsys):
    parser = argparse.ArgumentParser(prog="prog")
    parser.add_argument("--model_name")
    p = tmp_path / "cfg.yaml"
    # "nope" has no matching argparse dest -> must be rejected.
    p.write_text("model_name: foo\nnope: 1\n")
    with pytest.raises(SystemExit) as ei:
        apply_config_defaults(parser, str(p))
    assert ei.value.code == 2  # argparse.ArgumentParser.error() exits with status 2
    err = capsys.readouterr().err
    assert "unknown config key" in err
    assert "nope" in err  # names the offending key; a bare re-raise wouldn't
