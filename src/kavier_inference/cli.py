"""
Kavier Inference CLI - Main entry point for kavier-perf command.
"""

from __future__ import annotations

from kavier_inference.core.args import parse_args
from kavier_inference.core.service import run_performance


def main() -> None:
    """Main entry point for kavier-perf command."""
    args = parse_args()
    run_performance(args)


def main_efficiency() -> None:
    """Deprecated entry point for the kavier-eff command (use kavier-energy)."""
    import sys

    print(
        "kavier-eff is deprecated and will be removed in a future release; use kavier-energy instead.",
        file=sys.stderr,
    )

    from kavier_energy.calculator import main as energy_main

    energy_main()


if __name__ == "__main__":
    main()
