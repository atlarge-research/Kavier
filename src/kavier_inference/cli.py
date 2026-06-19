"""Console-script entry points for the inference simulator (``kavier-perf`` and the deprecated
``kavier-eff``/``kavier-energy``)."""

from __future__ import annotations

import sys

from kavier_inference.core.args import parse_args
from kavier_inference.core.service import run_performance
from kavier_library.lookup import UnknownSpecError


def main() -> None:
    """``kavier-perf`` entry point: parse CLI args and run the performance simulation; exits 2 on an
    unknown LLM/GPU spec."""
    args = parse_args()
    try:
        run_performance(args)
    except UnknownSpecError as exc:
        print(f"kavier-perf: error: {exc}", file=sys.stderr)
        sys.exit(2)


def main_efficiency() -> None:
    """``kavier-eff``: the efficiency entry point. Currently delegates to the energy
    calculator (the same engine as ``kavier-energy``); kept as a first-class command
    for the efficiency workflow."""
    from kavier_energy.calculator import main as energy_main

    energy_main()


if __name__ == "__main__":
    main()
