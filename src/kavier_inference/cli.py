"""Console-script entry points for the inference simulator (``kavier-perf`` and the deprecated ``kavier-eff``)."""

from __future__ import annotations

import sys

from kavier_inference.core.args import parse_args
from kavier_inference.core.service import run_performance
from kavier_library.lookup import UnknownSpecError


def main() -> None:
    """``kavier-perf`` entry point; exits 2 on an unknown LLM/GPU spec."""
    args = parse_args()
    try:
        run_performance(args)
    except UnknownSpecError as exc:
        print(f"kavier-perf: error: {exc}", file=sys.stderr)
        sys.exit(2)


def main_efficiency() -> None:
    """``kavier-eff``: delegates to the energy calculator (same engine as ``kavier-energy``)."""
    from kavier_energy.calculator import main as energy_main

    energy_main()


if __name__ == "__main__":
    main()
