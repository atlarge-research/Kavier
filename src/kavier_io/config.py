"""Tiny YAML run-config loader shared by the Kavier CLIs.

A config file is a flat mapping of ``{arg_name: value}`` that a CLI applies as argparse
defaults (via ``parser.set_defaults``) before parsing, so explicit flags still override it.
"""

from __future__ import annotations

import argparse
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Read a YAML mapping of ``{arg_name: value}`` from ``path``.

    Raises ``ValueError`` if the document is not a mapping (e.g. a list or scalar), so callers
    get a clear message instead of a confusing failure when they hand it the wrong file.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path!r} must be a YAML mapping of arg_name: value, got {type(data).__name__}")
    return data


def apply_config_defaults(parser: argparse.ArgumentParser, path: str) -> None:
    """Load the YAML config at ``path`` and apply it as defaults on ``parser``.

    Every key must name a real argument of ``parser`` (its ``dest``); unknown keys raise
    ``parser.error`` listing them. Applies the values via ``parser.set_defaults`` so they act
    as defaults that an explicit CLI flag still overrides. Call this BEFORE ``parse_args``.
    """
    values = load_config(path)
    valid = {action.dest for action in parser._actions if action.dest != "help"}
    unknown = sorted(k for k in values if k not in valid)
    if unknown:
        parser.error(f"unknown config key(s) in {path!r}: {', '.join(unknown)} (valid: {', '.join(sorted(valid))})")
    parser.set_defaults(**values)
