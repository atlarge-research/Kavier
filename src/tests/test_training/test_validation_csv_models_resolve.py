from pathlib import Path

import pandas as pd
import pytest

from kavier_library.gpu import GPU_SPEC_LIBRARY
from kavier_library.llm import LLM_SPEC_LIBRARY

from .conftest import simulatable_mask

SRC = Path(__file__).resolve().parent.parent.parent
CSV = SRC / "kavier_training" / "data" / "input" / "validation_clean.csv"


@pytest.mark.skipif(not CSV.exists(), reason="validation_clean.csv not present")
def test_validation_csv_simulatable_rows_resolve():
    df = pd.read_csv(CSV)
    sim = df.loc[simulatable_mask(df)]
    assert len(sim) > 0, "expected at least one simulatable row in validation_clean.csv"

    missing_models = sorted(set(sim["model_name"].unique()) - set(LLM_SPEC_LIBRARY))
    missing_gpus = sorted(set(sim["gpu_model"].unique()) - set(GPU_SPEC_LIBRARY))
    assert not missing_models, f"Missing LLM keys: {missing_models}"
    assert not missing_gpus, f"Missing GPU keys: {missing_gpus}"
