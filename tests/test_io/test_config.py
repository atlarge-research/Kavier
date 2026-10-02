"""Unit tests for the shared YAML run-config loader (``kavier.sdk.io.config``)."""

from __future__ import annotations

import argparse

import pytest

from kavier.sdk.io.config import config_argv, load_config


def test_load_config_parses_yaml_scalar_types(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model_name: mistral-7b-v0.1\nbatch_size: 4\n")
    result = load_config(str(p))
    # YAML decodes an unquoted 4 to int and an unquoted word to str.
    assert result == {"model_name": "mistral-7b-v0.1", "batch_size": 4}
    assert isinstance(result["batch_size"], int)


def test_load_config_empty_file_is_empty_mapping(tmp_path):
    # yaml.safe_load("") returns None; the loader maps it to {}.
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_config(str(p)) == {}


@pytest.mark.parametrize(
    ("body", "typename"),
    [("- a\n- b\n", "list"), ("42\n", "int"), ("just-a-string\n", "str")],
)
def test_load_config_rejects_non_mapping(tmp_path, body, typename):
    # Only a top-level mapping is valid; a sequence or scalar raises ValueError naming its type.
    p = tmp_path / "bad.yaml"
    p.write_text(body)
    with pytest.raises(ValueError) as ei:
        load_config(str(p))
    msg = str(ei.value)
    assert "must be a YAML mapping" in msg
    assert typename in msg


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prog")
    parser.add_argument("--model_name")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--method", choices=["lora", "full"])
    parser.add_argument("--trace", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-training", action="store_true")
    mode.add_argument("--powersource")
    return parser


def _cfg(tmp_path, body: str) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(body)
    return str(p)


def _parse(tmp_path, body: str, argv: list[str]) -> argparse.Namespace:
    parser = _parser()
    return parser.parse_args(config_argv(parser, _cfg(tmp_path, body), argv))


def test_config_values_are_parsed_like_flags(tmp_path):
    args = _parse(tmp_path, "model_name: foo\nbatch_size: 7\ntrace: t.parquet\nfrom_training: true\n", [])
    assert (args.model_name, args.batch_size, args.trace, args.from_training) == ("foo", 7, "t.parquet", True)


def test_command_line_flag_beats_config_value(tmp_path):
    args = _parse(tmp_path, "batch_size: 7\ntrace: t.parquet\n", ["--batch_size", "1", "--from-training"])
    assert args.batch_size == 1


def test_config_value_satisfies_required_flag_and_group(tmp_path):
    # --trace and the mode group are required; the config supplies both.
    args = _parse(tmp_path, "trace: t.parquet\npowersource: ps.parquet\n", [])
    assert (args.trace, args.powersource, args.from_training) == ("t.parquet", "ps.parquet", False)


def test_command_line_group_member_beats_config_group_member(tmp_path):
    # The config picks --powersource, the command line picks --from-training; the command line wins.
    args = _parse(tmp_path, "trace: t.parquet\npowersource: ps.parquet\n", ["--from-training"])
    assert args.from_training is True
    assert args.powersource is None


def test_config_value_overridden_on_command_line_is_not_checked(tmp_path):
    args = _parse(tmp_path, "method: bogus\ntrace: t.parquet\n", ["--method", "lora", "--from-training"])
    assert args.method == "lora"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("method: bogus\n", "invalid choice"),
        ("batch_size: 4.5\n", "invalid int value"),
        ("batch_size: [1, 2]\n", "batch_size"),
        ("from_training: maybe\n", "from_training"),
    ],
)
def test_bad_config_value_is_rejected_and_names_the_file(tmp_path, capsys, body, expected):
    parser = _parser()
    path = _cfg(tmp_path, body)
    with pytest.raises(SystemExit) as ei:
        config_argv(parser, path, ["--trace", "t.parquet"])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert expected in err
    assert path in err


def test_config_cannot_set_two_members_of_one_group(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse(tmp_path, "trace: t.parquet\nfrom_training: true\npowersource: ps.parquet\n", [])
    assert "not allowed with" in capsys.readouterr().err


def test_bad_command_line_value_is_reported_before_group_handling(tmp_path, capsys):
    # The config picks --powersource and the command line picks --from-training with a bad --batch_size.
    # The error is about the bad value; a group conflict would point at a mistake the user did not make.
    with pytest.raises(SystemExit):
        _parse(tmp_path, "trace: t.parquet\npowersource: ps.parquet\n", ["--from-training", "--batch_size", "x"])
    err = capsys.readouterr().err
    assert "invalid int value: 'x'" in err
    assert "not allowed with" not in err


def test_false_or_null_config_value_adds_no_flag(tmp_path):
    parser = _parser()
    path = _cfg(tmp_path, "from_training: false\npowersource: null\n")
    assert config_argv(parser, path, ["--trace", "t.parquet"]) == ["--trace", "t.parquet"]


def test_unknown_config_key_aborts(tmp_path, capsys):
    parser = _parser()
    # "nope" has no matching argparse dest.
    path = _cfg(tmp_path, "model_name: foo\nnope: 1\n")
    with pytest.raises(SystemExit) as ei:
        config_argv(parser, path, [])
    assert ei.value.code == 2  # ArgumentParser.error() exits with status 2
    err = capsys.readouterr().err
    assert "unknown config key" in err
    assert "nope" in err
