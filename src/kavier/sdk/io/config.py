"""YAML run-config loader: flat {arg_name: value} applied as argparse defaults, so explicit flags still override."""

from __future__ import annotations

import argparse
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Read a YAML {arg_name: value} mapping; ValueError if not a mapping."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path!r} must be a YAML mapping of arg_name: value, got {type(data).__name__}")
    return data


def apply_config_defaults(parser: argparse.ArgumentParser, path: str) -> None:
    """Apply YAML config as parser defaults (call BEFORE parse_args); unknown keys (non-dest) raise parser.error."""
    values = load_config(path)
    valid = {action.dest for action in parser._actions if action.dest != "help"}
    unknown = sorted(k for k in values if k not in valid)
    if unknown:
        parser.error(f"unknown config key(s) in {path!r}: {', '.join(unknown)} (valid: {', '.join(sorted(valid))})")
    parser.set_defaults(**values)
