"""
Kavier Inference CLI - Main entry point for kavier-perf command.
"""
from __future__ import annotations

import argparse

from kavier_inference.core.args import parse_args
from kavier_inference.core.service import run_performance


def main() -> None:
    """Main entry point for kavier-perf command."""
    args = parse_args()
    run_performance(args)


def main_efficiency() -> None:
    """Main entry point for kavier-eff command."""
    from efficiency.calculator import main as eff_main
    eff_main()


if __name__ == "__main__":
    main()

