"""``kavier calibrate`` subcommand: fit a training-calibration table from a profiling data file.

This exposes the dev-only from-scratch calibration fit (the two-tier regularized-Powell + per-cell
interaction_scale recipe that produces calibration.json) as a parameterized command, so it can be
driven from an arbitrary profiling CSV rather than the fixed internal trace. It is the backend for
Coastline's ``coastline-tune --method kavier``.

    kavier calibrate <input.csv> [--output PATH] [--models m1,m2,...]

``<input.csv>`` is a profiling trace with the fms-hf-tuning columns (model_name, gpu_model, method,
number_gpus, number_nodes, tokens_per_sample, batch_size, is_valid, dataset_tokens_per_second, ...).
The fit keeps only ``is_valid == 1`` and ``dataset_tokens_per_second > 0`` rows at ANY GPU count
(no <=8 cap) and targets ``dataset_tokens_per_second``. ``--models`` restricts the fit to a
comma-separated set; by default every model with enough valid rows in the file is fit. Any >8-GPU rows
in the input join the main fit directly (the multi-GPU correction is then fit jointly with everything
else); a sibling ``raw_trace.csv`` is only consulted when the input itself has no >8-GPU rows.

Output: the calibration JSON (same schema as calibration.json) is written to ``--output`` (default:
stdout). A summary -- models fitted, per-model row counts, and the held-out test MdAPE overall AND
broken down by model and by GPU-count -- goes to stderr, so ``kavier calibrate trace.csv > cal.json``
yields a clean file. If the dataset falls short of the suitability properties, a headline warning is
emitted first (the fit still runs on whatever the data supports).

Because the fit needs scipy/scikit-learn (the ``[calibration]`` extra), the engine is imported
lazily; without the extra the command prints an install hint and exits non-zero.

How Coastline consumes the output: point Kavier at the produced file before predicting, either by
env var ``KAVIER_CALIBRATION=<path>`` (read on first calibration access) or programmatically via
``kavier.sdk.training.calibration.use_calibration("<path>")``; subsequent
``kavier.training.performance(...)`` predictions then apply the fitted table.
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
    """Print the held-out test MdAPE broken down by MODEL and by total-GPU count on the SAME seed-42
    test rows (each line: ``MdAPE% (n=<test rows in that group>)``). ``evaluate`` is the engine's MdAPE
    function (rows, grad_accum_steps, backward_factor, cal_dict) -> MdAPE %; ``test`` is the test
    DataFrame (carrying model_name + the ``total`` = gpus*nodes column). Groups with no test rows never
    appear; an all-zero-measured group shows nan (evaluate's contract)."""
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

    # Summary (to stderr): which models were fit, their valid row counts, and an independently
    # recomputed held-out test MdAPE of the produced table on the SAME seed-42 test split the fit used
    # -- overall, then broken down by model and by total-GPU count (each as MdAPE% (n=<test rows>)).
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
