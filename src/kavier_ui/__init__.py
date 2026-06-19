"""Kavier interactive CLI — a guided, colourful REPL over the four simulators."""

from __future__ import annotations

__all__ = ["main"]


def main() -> None:
    from kavier_ui.app import main as _main

    _main()
