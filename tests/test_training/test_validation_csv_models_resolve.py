"""Tests for the validation-CSV simulatability filter (conftest.simulatable_mask).

Covers the four constraints of the mask (both library memberships, the total-GPU product cap,
throughput > 0), the throughput-column precedence, and, when the internal CSV is present, that every
row the mask keeps runs through the engine to a positive, finite throughput.
"""

import math
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pytest

from kavier.sdk.training.core.engine import simulate_full_training

from .conftest import simulatable_mask, throughput_column

# The (unvendored) internal validation CSV, if present, ships under the training package's data dir.
CSV = Path(str(files("kavier.sdk.training").joinpath("data", "input", "validation_clean.csv")))

# Two catalog keys defined in kavier/sdk/library/{llm,gpu}.py.
KNOWN_MODEL = "Llama-3-8B"
KNOWN_GPU = "A100-80GB"


def _base_row(**overrides: object) -> dict:
    row = {
        "model_name": KNOWN_MODEL,
        "gpu_model": KNOWN_GPU,
        "number_gpus": 1,
        "number_nodes": 1,
        "measured_throughput": 100.0,
    }
    row.update(overrides)
    return row


def test_mask_rejects_each_constraint_violation_independently():
    # Row 0 meets every constraint; rows 1-4 each break one, so only row 0 is kept.
    df = pd.DataFrame(
        [
            _base_row(),  # all-pass
            _base_row(model_name="UNKNOWN-MODEL"),  # not in LLM library
            _base_row(gpu_model="UNKNOWN-GPU"),  # not in GPU library
            _base_row(measured_throughput=0.0),  # throughput not > 0
            _base_row(measured_throughput=-5.0),  # negative throughput
        ]
    )
    assert list(simulatable_mask(df)) == [True, False, False, False, False]


def test_mask_total_gpu_cap_uses_product_and_is_inclusive_at_32():
    # total_gpus = number_gpus * number_nodes, capped at <= 32 (conftest MAX_TOTAL_GPUS).
    # 8*4 = 32 (in), 8*5 = 40 (out), 16*2 = 32 (in), 33*1 = 33 (out), 4*4 = 16 (in).
    # number_gpus alone would keep (8,5); a strict "< 32" would drop 32.
    df = pd.DataFrame(
        [
            _base_row(number_gpus=8, number_nodes=4),  # 32 -> in
            _base_row(number_gpus=8, number_nodes=5),  # 40 -> out
            _base_row(number_gpus=16, number_nodes=2),  # 32 -> in
            _base_row(number_gpus=33, number_nodes=1),  # 33 -> out
            _base_row(number_gpus=4, number_nodes=4),  # 16 -> in
        ]
    )
    assert list(simulatable_mask(df)) == [True, False, True, False, True]


def test_throughput_column_prefers_measured_then_actual_then_raises():
    both = pd.DataFrame({"measured_throughput": [1.0], "actual_throughput": [2.0]})
    only_actual = pd.DataFrame({"actual_throughput": [2.0]})
    neither = pd.DataFrame({"throughput": [3.0]})

    # measured_throughput wins when both exist; actual_throughput is the fallback
    assert throughput_column(both) == "measured_throughput"
    assert throughput_column(only_actual) == "actual_throughput"
    # neither column present raises
    with pytest.raises(KeyError):
        throughput_column(neither)


@pytest.mark.skipif(not CSV.exists(), reason="validation_clean.csv not present")
def test_validation_csv_simulatable_rows_run_to_positive_throughput():
    # Every (model, gpu) pair the mask keeps must run through the engine to a positive, finite
    # throughput. A renamed or removed catalog key (get_* raises) or a 0/NaN result fails.
    df = pd.read_csv(CSV)
    sim = df.loc[simulatable_mask(df)]
    assert len(sim) > 0, "expected at least one simulatable row in validation_clean.csv"

    # One row per distinct (model, gpu) pair covers every catalog key in the CSV.
    seen = sim.drop_duplicates(subset=["model_name", "gpu_model"])
    for row in seen.itertuples(index=False):
        pred = simulate_full_training(
            model_name=row.model_name,
            method=row.method,
            gpu_model=row.gpu_model,
            tokens_per_sample=int(row.tokens_per_sample),
            batch_size=int(row.batch_size),
            number_gpus=int(row.number_gpus),
            number_nodes=int(row.number_nodes),
        )["train_tokens_per_second"]
        assert math.isfinite(pred) and pred > 0, f"{row.model_name} on {row.gpu_model} -> {pred}"
