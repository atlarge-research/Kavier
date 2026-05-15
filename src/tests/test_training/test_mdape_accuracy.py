from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kavier_training.core.engine import simulate_full_training

from .conftest import simulatable_mask, throughput_column

SRC = Path(__file__).resolve().parent.parent.parent
VALIDATION_CSV = SRC / "kavier_training" / "data" / "input" / "validation_clean.csv"

MDAPE_THRESHOLD_PCT = 12.0
MIN_SAMPLES = 100


@pytest.mark.skipif(not VALIDATION_CSV.exists(), reason="validation_clean.csv not present")
def test_mdape_on_validation_clean():
    df = pd.read_csv(VALIDATION_CSV)
    sim = df.loc[simulatable_mask(df)].copy()
    assert len(sim) >= MIN_SAMPLES, f"need ≥{MIN_SAMPLES} simulatable rows, got {len(sim)}"

    tcol = throughput_column(sim)
    apes: list[float] = []

    for row in sim.itertuples(index=False):
        pred = simulate_full_training(
            model_name=row.model_name,
            method=row.method,
            gpu_model=row.gpu_model,
            tokens_per_sample=int(row.tokens_per_sample),
            batch_size=int(row.batch_size),
            number_gpus=int(row.number_gpus),
            number_nodes=int(row.number_nodes),
        )["train_tokens_per_second"]
        actual = float(getattr(row, tcol))
        if actual > 0 and pred > 0:
            apes.append(abs(pred - actual) / actual * 100.0)

    assert len(apes) >= MIN_SAMPLES, f"too few valid predictions: {len(apes)}"
    mdape = float(np.median(apes))
    assert mdape <= MDAPE_THRESHOLD_PCT, (
        f"MdAPE {mdape:.2f}% exceeds threshold {MDAPE_THRESHOLD_PCT}% (n={len(apes)} rows from validation_clean.csv)"
    )
