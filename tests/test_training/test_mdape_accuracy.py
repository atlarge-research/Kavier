"""Training-model accuracy layer: Median Absolute Percentage Error vs measured hardware.

The engine (``simulate_full_training``) is validated against an *independent* oracle -- real
measured training throughput recorded in the (unvendored) internal validation CSV. The MdAPE
metric that turns those two series into a single accuracy number is load-bearing for the 12%
contract, so it is extracted here and unit-tested with hand-derived oracles that always run,
even on a clean checkout where the validation CSV is absent and the accuracy test skips.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kavier.sdk.training.core.engine import simulate_full_training

from .conftest import simulatable_mask, throughput_column

# The (unvendored) internal validation CSV, if present, ships under the training package's data dir.
VALIDATION_CSV = Path(str(files("kavier.sdk.training").joinpath("data", "input", "validation_clean.csv")))

# Published accuracy ceiling for the calibrated training model (docs/content/known-weaknesses.md).
MDAPE_THRESHOLD_PCT = 12.0
MIN_SAMPLES = 100


def _median_ape_pct(predictions: Sequence[float], actuals: Sequence[float]) -> tuple[float, int]:
    """Median absolute percentage error (in %) over pairs where both values are strictly positive.

    Returns ``(mdape_pct, n_valid)``. Non-positive predictions/actuals are dropped (a <=0 measured
    throughput is a bad data row; a <=0 prediction is a degenerate engine output that must not
    silently define an APE of 0).
    """
    apes = [abs(p - a) / a * 100.0 for p, a in zip(predictions, actuals, strict=True) if a > 0 and p > 0]
    return float(np.median(apes)), len(apes)


def test_median_ape_matches_hand_computed_percentages() -> None:
    # preds/actuals chosen so APEs are trivial to derive by hand:
    #   |150-100|/100*100 = 50 ; |90-100|/100*100 = 10 ; |100-100|/100*100 = 0
    # sorted -> [0, 10, 50]; the median of 3 values is the middle element -> 10.0
    mdape, n = _median_ape_pct([150.0, 90.0, 100.0], [100.0, 100.0, 100.0])
    assert (mdape, n) == (10.0, 3)


def test_median_ape_averages_the_two_middle_values() -> None:
    # Even count pins that we use a true median (mean of the two middle), not sorted[n//2].
    #   APEs: 10, 20, 30, 40 -> sorted middle pair (20, 30) -> mean 25.0
    # A sorted[n//2] implementation would return 30.0 and fail here.
    mdape, n = _median_ape_pct([110.0, 120.0, 130.0, 140.0], [100.0, 100.0, 100.0, 100.0])
    assert (mdape, n) == (25.0, 4)


def test_median_ape_drops_nonpositive_rows() -> None:
    # Only the first pair is valid: (5, 0) drops on actual<=0, (-3, 50) drops on pred<=0,
    # (100, -10) drops on actual<=0. Remaining APE = |150-100|/100*100 = 50 ; n = 1.
    # If the filter were removed, (5, 0) would divide by zero -> inf and blow up the median.
    mdape, n = _median_ape_pct([150.0, 5.0, -3.0, 100.0], [100.0, 0.0, 50.0, -10.0])
    assert (mdape, n) == (50.0, 1)


@pytest.mark.skipif(not VALIDATION_CSV.exists(), reason="validation_clean.csv not present")
def test_mdape_on_validation_clean() -> None:
    # Oracle: measured training throughput from real hardware runs (independent of the engine).
    # Contract: the calibrated engine predicts within MDAPE_THRESHOLD_PCT of measured on the
    # supported model/GPU set. A `return <constant>` engine, or a unit/scaling regression (e.g.
    # the /1e9 vs /1e12 FLOPs base), pushes the median APE far past 12% -> red.
    df = pd.read_csv(VALIDATION_CSV)
    sim = df.loc[simulatable_mask(df)].copy()
    assert len(sim) >= MIN_SAMPLES, f"need >={MIN_SAMPLES} simulatable rows, got {len(sim)}"

    tcol = throughput_column(sim)
    preds: list[float] = []
    actuals: list[float] = []
    for row in sim.itertuples(index=False):
        preds.append(
            simulate_full_training(
                model_name=row.model_name,
                method=row.method,
                gpu_model=row.gpu_model,
                tokens_per_sample=int(row.tokens_per_sample),
                batch_size=int(row.batch_size),
                number_gpus=int(row.number_gpus),
                number_nodes=int(row.number_nodes),
            )["train_tokens_per_second"]
        )
        actuals.append(float(getattr(row, tcol)))

    mdape, n = _median_ape_pct(preds, actuals)
    # Guard the accuracy claim: a regression that zeroes/NaNs half the predictions would silently
    # shrink the sample instead of raising the error, so require enough valid pairs to survive.
    assert n >= MIN_SAMPLES, f"too few valid predictions: {n}"
    assert mdape <= MDAPE_THRESHOLD_PCT, (
        f"MdAPE {mdape:.2f}% exceeds threshold {MDAPE_THRESHOLD_PCT}% (n={n} rows from validation_clean.csv)"
    )
