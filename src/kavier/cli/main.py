"""The unified ``kavier`` command-line interface: one entrypoint, six subcommands.

    kavier inference ...   run the per-request inference simulator
    kavier training ...    run the analytical training simulator
    kavier cluster ...     simulate a FIFO/backfill GPU cluster of jobs with known durations
    kavier energy ...      per-Mtoken energy/$ efficiency
    kavier carbon ...      CO2 vs a carbon trace
    kavier calibrate ...   fit a training-calibration table from a profiling CSV ([calibration] extra)

Each subcommand delegates to its engine's own parser, so ``kavier <cmd> --help`` shows that command's
flags. The interactive REPL is a separate entrypoint (``kavier-ui`` / ``python -m kavier.ui``).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from kavier.sdk.domain import Domain

# subcommand -> (one-line help, submodule name under `kavier.cli`). The module is imported lazily at dispatch
# time (see _run_subcommand), so `kavier --help` and `kavier <cmd>` never pull a sibling command's heavy
# dependencies (pandas/numpy load only on the run path).
_COMMANDS: dict[str, tuple[str, str]] = {
    Domain.INFERENCE: ("Run the per-request inference simulator (latency/throughput + OpenDC export).", "inference"),
    Domain.TRAINING: ("Run the analytical training simulator (throughput/runtime).", "training"),
    "cluster": ("Simulate a FIFO/backfill GPU cluster running jobs of known duration.", "cluster"),
    "energy": ("Per-Mtoken energy/$ efficiency from Kavier + OpenDC output.", "energy"),
    "carbon": ("Estimate CO2 from a training sim or OpenDC power against a carbon trace.", "carbon"),
    "calibrate": ("Fit a training-calibration table from a profiling CSV ([calibration] extra).", "calibrate"),
}


def _run_subcommand(module: str, argv: Sequence[str] | None) -> None:
    """Import ``kavier.cli.<module>`` lazily and call its ``main`` — sibling deps stay off the load path."""
    importlib.import_module(f"kavier.cli.{module}").main(argv)


def _build_root_parser() -> argparse.ArgumentParser:
    from kavier import __version__

    parser = argparse.ArgumentParser(
        prog="kavier",
        description="Kavier — simulate performance, sustainability, and efficiency of LLM ecosystems.",
        epilog="Run 'kavier <command> --help' for command-specific options. For the interactive UI, use 'kavier-ui'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=f"kavier {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{" + ",".join(_COMMANDS) + "}")
    for name, (help_text, _module) in _COMMANDS.items():
        # add_help=False: each command's real flags live in its engine parser (kavier <cmd> --help delegates there).
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch ``kavier <command> ...`` to the matching engine, or print help/version."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_root_parser()

    if not argv:  # bare `kavier`: show help (exit 2, git-style)
        parser.print_help(sys.stderr)
        raise SystemExit(2)
    if argv[0] in ("-h", "--help", "-V", "--version"):
        parser.parse_args(argv)  # prints help/version and exits 0
        return

    command, rest = argv[0], argv[1:]
    entry = _COMMANDS.get(command)
    if entry is None:
        # ``map(str, ...)`` renders the two Domain-enum keys as their plain values, so the choice list
        # is repr'd identically to the all-string dict (``'inference'`` not ``<Domain.INFERENCE: ...>``).
        valid = ", ".join(map(repr, map(str, _COMMANDS)))
        parser.error(f"argument command: invalid choice: {command!r} (choose from {valid})")
    _run_subcommand(entry[1], rest)


if __name__ == "__main__":
    main()
