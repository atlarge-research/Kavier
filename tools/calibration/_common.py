"""Shared helpers for the DEV-ONLY calibration fitters in this directory.

These scripts refit Kavier's empirical (Tier-2) calibration terms from IBM-internal
ado-sfttrainer / PD1 profiling traces. They are NOT part of the installed package and
are never shipped; importing this module makes the kavier source tree importable from
``tools/calibration/`` so the fitters can run against the in-repo sources.

What it factors out (previously copy-pasted across all three fitters):
  * the ``sys.path`` bootstrap that puts ``<repo>/src`` (and the sibling ``coastline``
    checkout) on the path so ``import kavier_training`` / ``from trainer.common`` resolve;
  * :data:`CAL_PATH` — the shipped ``calibration.json`` location;
  * :func:`mdape` — median absolute percentage error;
  * :func:`calibration_override` — a context manager that swaps the live calibration
    (``kavier_training.core.calibration._CAL``) and restores it afterwards;
  * :func:`base_arg_parser` — the common ``--trace`` / ``--write`` argparse setup.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap. This file lives at <repo>/tools/calibration/_common.py, so:
#   parents[0] = tools/calibration
#   parents[1] = tools
#   parents[2] = <repo root>            (the kavier checkout)
#   parents[3] = <workspace>            (parent dir holding kavier, coastline, ...)
# The fitters import this module first, which makes the kavier sources importable.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
SRC = REPO_ROOT / "src"
WORKSPACE = _HERE.parents[3]
COASTLINE = WORKSPACE / "coastline"
TRACE_ARCHIVE = WORKSPACE / "trace-archive" / "pd1-profiling-dataset"

for _p in (str(SRC), str(COASTLINE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CAL_PATH = SRC / "kavier_training" / "data" / "calibration.json"


def mdape(measured: np.ndarray, pred: np.ndarray) -> float:
    """Median absolute percentage error (%), over rows with ``measured > 0``.

    Callers that already pre-filter to positive measured throughput see the mask as
    a no-op, so results are identical to the un-masked ``median(|p-y|/y)*100`` form.
    """
    m = measured > 0
    return float(np.median(np.abs((pred[m] - measured[m]) / measured[m])) * 100.0)


@contextmanager
def calibration_override(cal_dict: dict[str, Any]) -> Iterator[None]:
    """Temporarily install ``cal_dict`` as the live calibration, then restore.

    The engine reads ``kavier_training.core.calibration._CAL`` at simulate time, so a
    fit/eval pass that wants different scales swaps it for the duration of the block:

        with calibration_override(neutral_cal):
            preds = [simulate_training_step(...) for ...]
    """
    import kavier_training.core.calibration as cal

    saved = cal._CAL
    cal._CAL = cal_dict
    try:
        yield
    finally:
        cal._CAL = saved


def base_arg_parser(description: str | None) -> argparse.ArgumentParser:
    """Argparse parser with the shared ``--trace`` / ``--write`` flags.

    Individual fitters add their own extra flags (``--out``, ``--models``, ``--counts``)
    and set the ``--trace`` default to the trace they expect.
    """
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trace", type=Path, help="profiling CSV (IBM-internal; not vendored)")
    ap.add_argument("--write", action="store_true", help="write the fitted values into calibration.json")
    return ap
