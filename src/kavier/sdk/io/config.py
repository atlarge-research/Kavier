"""YAML run-config loader: a flat {arg_name: value} mapping turned into command-line flags.

The flags go in front of the typed arguments, so argparse checks their type and choices as if typed,
they count toward required flags, and a flag typed on the command line wins.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Sequence
from typing import Any, NoReturn

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Read a YAML {arg_name: value} mapping; raise ValueError if the file holds anything else."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path!r} must be a YAML mapping of arg_name: value, got {type(data).__name__}")
    return data


class _ProbeError(Exception):
    """A parse error from ``_Probe``."""


class _Probe(argparse.ArgumentParser):
    """Parser that raises ``_ProbeError`` where ``ArgumentParser`` would print and exit."""

    def error(self, message: str) -> NoReturn:
        raise _ProbeError(message)


def _probe_for(parser: argparse.ArgumentParser) -> _Probe:
    """Return a copy of ``parser``'s options with no defaults, nothing required and no help flag."""
    probe = _Probe(add_help=False, prefix_chars=parser.prefix_chars, allow_abbrev=parser.allow_abbrev)
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._VersionAction)):
            continue
        clone = copy.copy(action)
        clone.default = argparse.SUPPRESS
        clone.required = False
        probe._add_action(clone)
    return probe


def _typed_dests(probe: _Probe, argv: Sequence[str]) -> set[str]:
    """Return the dests that ``argv`` sets; raise ``_ProbeError`` if a value in ``argv`` is invalid."""
    namespace, _ = probe.parse_known_args(list(argv))
    return set(vars(namespace))


def _as_flags(action: argparse.Action, value: object) -> list[str]:
    """Return the command-line tokens for one config value; raise ValueError if no flag can express it."""
    flag = max(action.option_strings, key=len)
    if value is None:
        return []
    if action.nargs == 0:  # store_true / store_false
        if not isinstance(value, bool):
            raise ValueError(f"must be true or false, got {value!r}")
        return [flag] if value == action.const else []
    if isinstance(value, (list, dict)):
        raise ValueError(f"must be a single value, got {value!r}")
    return [f"{flag}={value}"]


def config_argv(parser: argparse.ArgumentParser, path: str, argv: Sequence[str]) -> list[str]:
    """Return ``argv`` with the config file's values put in front as flags; pass the result to parse_args.

    Keys set in ``argv``, and keys whose mutually exclusive group has a member in ``argv``, are left out.
    An unknown key or a value that fails the flag's type or choices calls ``parser.error``.
    """
    values = load_config(path)
    actions = {a.dest: a for a in parser._actions if a.option_strings and a.dest != "help"}
    unknown = sorted(k for k in values if k not in actions)
    if unknown:
        parser.error(f"unknown config key(s) in {path!r}: {', '.join(unknown)} (valid: {', '.join(sorted(actions))})")

    probe = _probe_for(parser)
    try:
        typed = _typed_dests(probe, argv)
    except _ProbeError as exc:  # the probe has no required flags or groups, so this is an argv error
        parser.error(str(exc))
    skip = set(typed)
    for group in parser._mutually_exclusive_groups:
        members = {a.dest for a in group._group_actions}
        if members & typed:
            skip |= members

    tokens: list[str] = []
    for key, value in values.items():
        if key in skip:
            continue
        try:
            tokens += _as_flags(actions[key], value)
        except ValueError as exc:
            parser.error(f"config file {path!r}: {key} {exc}")
    try:
        probe.parse_known_args(tokens)
    except _ProbeError as exc:
        parser.error(f"config file {path!r}: {exc}")
    return [*tokens, *argv]
