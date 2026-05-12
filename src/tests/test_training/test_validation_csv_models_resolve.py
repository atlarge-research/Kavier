"""Every model_name and gpu_model in bundled validation_results.csv must resolve."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from library.gpu import GPU_SPEC_LIBRARY  # noqa: E402
from library.llm import LLM_SPEC_LIBRARY  # noqa: E402

CSV = SRC / "kavier_training" / "data" / "output" / "validation_results.csv"


@pytest.mark.skipif(not CSV.exists(), reason="validation_results.csv not present")
def test_validation_csv_model_and_gpu_keys_exist():
    df = pd.read_csv(CSV)
    missing_models = sorted(set(df["model_name"].unique()) - set(LLM_SPEC_LIBRARY))
    missing_gpus = sorted(set(df["gpu_model"].unique()) - set(GPU_SPEC_LIBRARY))
    assert not missing_models, f"Missing LLM keys: {missing_models}"
    assert not missing_gpus, f"Missing GPU keys: {missing_gpus}"
