"""Training-model accuracy: median absolute percentage error (MdAPE) against measured hardware.

``simulate_full_training`` is compared with measured training throughput from the unvendored internal
validation CSV. The MdAPE helper behind the 12% threshold is unit-tested on hand-computed inputs, so
those tests run on a clean checkout where the CSV is absent and the accuracy test skips.
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

# Published accuracy ceiling for the calibrated training model [%].
MDAPE_THRESHOLD_PCT = 12.0
MIN_SAMPLES = 100


def _median_ape_pct(predictions: Sequence[float], actuals: Sequence[float]) -> tuple[float, int]:
    """Return the median absolute percentage error [%] over pairs where both values are positive.

    Returns ``(mdape_pct, n_valid)``. A measured throughput <= 0 is a bad data row; a prediction <= 0
    is a degenerate engine output. Both are dropped.
    """
    apes = [abs(p - a) / a * 100.0 for p, a in zip(predictions, actuals, strict=True) if a > 0 and p > 0]
    return float(np.median(apes)), len(apes)


def test_median_ape_matches_hand_computed_percentages() -> None:
    #   |150-100|/100*100 = 50 ; |90-100|/100*100 = 10 ; |100-100|/100*100 = 0
    # sorted -> [0, 10, 50]; the median of 3 values is the middle element -> 10.0
    mdape, n = _median_ape_pct([150.0, 90.0, 100.0], [100.0, 100.0, 100.0])
    assert (mdape, n) == (10.0, 3)


def test_median_ape_averages_the_two_middle_values() -> None:
    # Even count: the median is the mean of the middle pair.
    #   APEs: 10, 20, 30, 40 -> middle pair (20, 30) -> 25.0; sorted[n//2] would give 30.0.
    mdape, n = _median_ape_pct([110.0, 120.0, 130.0, 140.0], [100.0, 100.0, 100.0, 100.0])
    assert (mdape, n) == (25.0, 4)


def test_median_ape_drops_nonpositive_rows() -> None:
    # Only the first pair is valid: (5, 0) drops on actual<=0, (-3, 50) drops on pred<=0,
    # (100, -10) drops on actual<=0. Remaining APE = |150-100|/100*100 = 50 ; n = 1.
    # Without the filter (5, 0) would divide by zero.
    mdape, n = _median_ape_pct([150.0, 5.0, -3.0, 100.0], [100.0, 0.0, 50.0, -10.0])
    assert (mdape, n) == (50.0, 1)


@pytest.mark.skipif(not VALIDATION_CSV.exists(), reason="validation_clean.csv not present")
def test_mdape_on_validation_clean() -> None:
    # Measured throughput from real hardware runs. The calibrated engine must predict within
    # MDAPE_THRESHOLD_PCT on the supported model/GPU set; a constant engine or a unit error such as
    # /1e9 vs /1e12 for the FLOPs base pushes the MdAPE well past 12%.
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
    # Zero or NaN predictions are dropped and would shrink the sample without raising the error,
    # so require enough valid pairs.
    assert n >= MIN_SAMPLES, f"too few valid predictions: {n}"
    assert mdape <= MDAPE_THRESHOLD_PCT, (
        f"MdAPE {mdape:.2f}% exceeds threshold {MDAPE_THRESHOLD_PCT}% (n={n} rows from validation_clean.csv)"
    )
