"""The ``kavier`` command-line interface with six subcommands.

    kavier inference ...   run the per-request inference simulator
    kavier training ...    run the analytical training simulator
    kavier cluster ...     simulate a FIFO/backfill GPU cluster of jobs with known durations
    kavier energy ...      per-Mtoken energy/$ efficiency
    kavier carbon ...      CO2 vs a carbon trace
    kavier calibrate ...   fit a training-calibration table from a profiling CSV ([calibration] extra)

Each subcommand has its own parser, so ``kavier <cmd> --help`` shows that command's flags.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from kavier.sdk.domain import Domain

# subcommand -> (help, submodule of `kavier.cli`). Imported at dispatch, so pandas/numpy load only
# for the command that runs.
_COMMANDS: dict[str, tuple[str, str]] = {
    Domain.INFERENCE: ("Run the per-request inference simulator (latency/throughput + OpenDC export).", "inference"),
    Domain.TRAINING: ("Run the analytical training simulator (throughput/runtime).", "training"),
    "cluster": ("Simulate a FIFO/backfill GPU cluster running jobs of known duration.", "cluster"),
    "energy": ("Per-Mtoken energy/$ efficiency from Kavier + OpenDC output.", "energy"),
    "carbon": ("Estimate CO2 from a training sim or OpenDC power against a carbon trace.", "carbon"),
    "calibrate": ("Fit a training-calibration table from a profiling CSV ([calibration] extra).", "calibrate"),
}


def _run_subcommand(module: str, argv: Sequence[str] | None) -> None:
    """Import ``kavier.cli.<module>`` and call its ``main``."""
    importlib.import_module(f"kavier.cli.{module}").main(argv)


def _build_root_parser() -> argparse.ArgumentParser:
    from kavier import __version__

    parser = argparse.ArgumentParser(
        prog="kavier",
        description="Kavier: simulate performance, sustainability, and efficiency of LLM ecosystems.",
        epilog="Run 'kavier <command> --help' for command-specific options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=f"kavier {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{" + ",".join(_COMMANDS) + "}")
    for name, (help_text, _module) in _COMMANDS.items():
        # The command's own parser handles --help.
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch ``kavier <command> ...`` to the matching engine, or print help/version."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_root_parser()

    if not argv:  # bare `kavier`: print help and exit 2, as git does
        parser.print_help(sys.stderr)
        raise SystemExit(2)
    if argv[0] in ("-h", "--help", "-V", "--version"):
        parser.parse_args(argv)  # prints help/version and exits 0
        return

    command, rest = argv[0], argv[1:]
    entry = _COMMANDS.get(command)
    if entry is None:
        # str() turns the Domain keys into plain values, e.g. 'inference'.
        valid = ", ".join(map(repr, map(str, _COMMANDS)))
        parser.error(f"argument command: invalid choice: {command!r} (choose from {valid})")
    _run_subcommand(entry[1], rest)


if __name__ == "__main__":
    main()
