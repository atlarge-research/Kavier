"""`python -m kavier.ui` entry point."""

from __future__ import annotations

import sys

from kavier.ui.app import main
from kavier.ui.prompts import Abort
from kavier.ui.theme import console

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, Abort):
        console.print("\n[cyan]  bye 👋[/]\n")
        sys.exit(0)
