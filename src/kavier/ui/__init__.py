"""Kavier interactive CLI — a guided, colourful REPL over the four simulators."""

from __future__ import annotations

import sys

__all__ = ["main"]


def main() -> None:
    from kavier.ui.app import main as _main
    from kavier.ui.prompts import Abort
    from kavier.ui.theme import console

    try:
        _main()
    except (KeyboardInterrupt, Abort):
        console.print("\n[cyan]  bye 👋[/]\n")
        sys.exit(0)
