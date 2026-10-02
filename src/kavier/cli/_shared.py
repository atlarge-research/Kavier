"""Argparse helpers for the Kavier subcommands.

``FriendlyParser`` adds a worked example to error messages. ``parse_args_with_config`` reads
``--config`` and puts the YAML values in front of the typed arguments as flags, so argparse checks them
and typed flags win. The fold itself is in ``kavier.sdk.io.config``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import NoReturn

from kavier.sdk.io.config import config_argv


class FriendlyParser(argparse.ArgumentParser):
    """``ArgumentParser`` whose error message appends a worked example (set via ``example=``)."""

    def __init__(self, *args: object, example: str | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.example = example

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        if self.example:
            print(f"\nExample:\n  {self.example}", file=sys.stderr)
        sys.exit(2)


def peek_config(argv: Sequence[str] | None = None) -> str | None:
    """Return the ``--config`` value from ``argv`` without full parsing (``None`` if absent)."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", default=None)
    known, _ = peek.parse_known_args(argv)
    config: str | None = known.config
    return config


def parse_args_with_config(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` (default ``sys.argv[1:]``) with the values of its ``--config`` file, if any, as flags."""
    args = list(sys.argv[1:] if argv is None else argv)
    path = peek_config(args)
    if path is not None:
        args = config_argv(parser, path, args)
    return parser.parse_args(args)
