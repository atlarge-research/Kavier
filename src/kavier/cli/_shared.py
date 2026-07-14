"""Shared argparse plumbing for the Kavier subcommands: a ``FriendlyParser`` that appends a worked
example to error messages, and the ``--config`` YAML peek/fold (an explicit flag still overrides a
config value). The fold logic itself lives in ``kavier.sdk.io.config`` so both the engine CLIs and the
unified ``kavier`` CLI can import it without a layering inversion.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import NoReturn

from kavier.sdk.io.config import apply_config_defaults


class FriendlyParser(argparse.ArgumentParser):
    """``ArgumentParser`` whose error message appends a worked example (set via ``example=``)."""

    def __init__(self, *args: object, example: str | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.example = example

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        if self.example:
            print(f"\nYou may have mistaken the input — try this example instead:\n  {self.example}", file=sys.stderr)
        sys.exit(2)


def peek_config(argv: Sequence[str] | None = None) -> str | None:
    """Return the ``--config`` value from ``argv`` without full parsing (``None`` if absent)."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", default=None)
    known, _ = peek.parse_known_args(argv)
    config: str | None = known.config
    return config


def apply_config(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> None:
    """Fold a ``--config`` YAML file into ``parser`` defaults (call BEFORE ``parse_args``); no-op if absent."""
    path = peek_config(argv)
    if path is not None:
        apply_config_defaults(parser, path)
