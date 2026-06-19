"""`python -m kavier_ui` entry point."""
from __future__ import annotations

import sys

from kavier_ui.app import main
from kavier_ui.prompts import Abort
from kavier_ui.theme import console

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, Abort):
        console.print("\n[cyan]  bye 👋[/]\n")
        sys.exit(0)
