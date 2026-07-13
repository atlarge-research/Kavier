"""``kavier calibrate`` subcommand: fit a training-calibration table from a profiling CSV.

Exposes the dev-only from-scratch calibration fit (the two-tier Powell recipe behind
calibration.json) as a command, so it can run on an arbitrary profiling trace. This is the backend
for Coastline's ``coastline-tune --method kavier``.

    kavier calibrate <input.csv> [--output PATH] [--models m1,m2,...]

``<input.csv>`` carries the fms-hf-tuning columns (model_name, gpu_model, method, number_gpus,
number_nodes, tokens_per_sample, batch_size, is_valid, dataset_tokens_per_second, ...). Unlike the
shipped tables, the fit keeps valid rows at ANY GPU count (no <=8 cap), and if the input has no
>8-GPU rows it falls back to a sibling ``raw_trace.csv`` for them. ``--models`` restricts the fit;
by default every model with enough valid rows is fit.

The JSON goes to ``--output`` (default stdout); the fit summary and held-out test MdAPE go to
stderr, so ``kavier calibrate trace.csv > cal.json`` still yields a clean file. The fit needs the
``[calibration]`` extra (scipy/scikit-learn), imported lazily; without it the command exits
non-zero with an install hint.

Coastline then points Kavier at the output before predicting, via ``KAVIER_CALIBRATION=<path>`` or
``kavier.sdk.training.calibration.use_calibration("<path>")``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

_EXAMPLE_CMD = "kavier calibrate profiling_trace.csv --output cal.json --models granite-3-8b,mistral-7b-v0.1"

_MISSING_EXTRA_MSG = "kavier calibrate needs the [calibration] extra: uv sync --extra calibration"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kavier calibrate",
        description="Fit a training-calibration table from a profiling CSV (the from-scratch two-tier fit).",
        epilog=f"Example: {_EXAMPLE_CMD}",
    )
    parser.add_argument("input", help="Path to the profiling CSV to calibrate on")
    parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="Where to write the calibration JSON ('-' or omitted: stdout)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to fit (default: every model with enough valid rows in the file)",
    )
    return parser


def _print_regime_breakdown(test: pd.DataFrame, cal: dict[str, Any], evaluate: Callable[..., float]) -> None:
    """Print the held-out test MdAPE broken down by model and by total-GPU count over the same test
    rows. ``evaluate`` is the engine's MdAPE function; ``test`` carries model_name and the ``total`` =
    gpus*nodes column. An all-zero-measured group shows nan (evaluate's contract)."""
    print("  test MdAPE by model:", file=sys.stderr)
    for model in sorted(test["model_name"].astype(str).unique()):
        sub = test[test["model_name"].astype(str) == model]
        print(f"    {model:<26} {evaluate(sub, 1, 2.0, cal):>7.2f}%  (n={len(sub)})", file=sys.stderr)

    print("  test MdAPE by GPU-count (total_gpus = number_gpus * number_nodes):", file=sys.stderr)
    for count in sorted(int(t) for t in test["total"].unique()):
        sub = test[test["total"] == count]
        print(f"    {count:>4} GPU  {evaluate(sub, 1, 2.0, cal):>7.2f}%  (n={len(sub)})", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> None:
    """Fit a calibration table from a profiling CSV and write the JSON to --output (default stdout)."""
    args = _build_parser().parse_args(argv)

    # Lazy, guarded import: engine.py pulls scipy/scikit-learn at module top, which are dev-only.
    try:
        from kavier.sdk.training.calibration.engine import (
            _dumps,
            _filter_valid_rows,
            calibrate,
            evaluate,
            train_val_test_split,
        )
    except ImportError:
        print(_MISSING_EXTRA_MSG, file=sys.stderr)
        raise SystemExit(1) from None

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None

    cal = calibrate(args.input, models)
    text = _dumps(cal)

    # Summary to stderr: models fit + valid row counts, then the held-out test MdAPE recomputed on the
    # same seed-42 test split the fit used (overall, then by model and by GPU count).
    import pandas as pd

    trace = pd.read_csv(args.input, low_memory=False)
    fitted = list(cal["model_scale"].keys())
    valid = _filter_valid_rows(trace, fitted, max_total_gpus=None)
    counts = valid["model_name"].astype(str).value_counts().to_dict()
    _, _, test = train_val_test_split(valid)
    held_out = evaluate(test, 1, 2.0, cal)

    print(f"calibrated {len(fitted)} model(s) on {len(valid)} valid rows (any GPU count):", file=sys.stderr)
    for model in fitted:
        print(f"  {model:<28} {int(counts.get(model, 0)):>6} rows", file=sys.stderr)
    print(f"held-out test MdAPE (overall): {held_out:.2f}%  (n={len(test)})", file=sys.stderr)

    _print_regime_breakdown(test, cal, evaluate)

    if args.output in (None, "-"):
        sys.stdout.write(text)
    else:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
