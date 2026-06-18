from __future__ import annotations

import sys

from kavier_inference.core.args import parse_args
from kavier_inference.core.service import run_performance
from kavier_library.lookup import UnknownSpecError


def main() -> None:
    args = parse_args()
    try:
        run_performance(args)
    except UnknownSpecError as exc:
        print(f"kavier-perf: error: {exc}", file=sys.stderr)
        sys.exit(2)


def main_efficiency() -> None:
    print(
        "kavier-eff is deprecated and will be removed in a future release; use kavier-energy instead.",
        file=sys.stderr,
    )

    from kavier_energy.calculator import main as energy_main

    energy_main()


if __name__ == "__main__":
    main()
