"""Tests for the kavier.sdk.training.calibration getters over the shipped calibration.json.

Covered: (1) each getter reads its JSON key and returns a float, (2) uncovered keys fall back to 1.0
with a one-time warning, (3) multi_gpu_correction is 1.0 for <=1 GPU and snaps absent counts to the
nearest fitted count without interpolation, (4) interaction_scale returns 1.0 for absent cells without
a warning or snapping. Expected values come from the JSON file read from disk.
"""

from __future__ import annotations

import copy
import json
import warnings
from enum import Enum
from pathlib import Path

import pytest

from kavier.sdk.training import calibration

# The shipped calibration.json read from disk: the table the getters use (via the autouse fixture)
# and the expected values for the key-mapping tests.
_CAL_PATH = Path(calibration.__file__).resolve().parent / "calibration.json"
with _CAL_PATH.open(encoding="utf-8") as _f:
    RAW = json.load(_f)


@pytest.fixture(autouse=True)
def _pin_shipped_calibration():
    """Pin the live table to a copy of the shipped JSON and clear the warned-key set, then restore both.

    test_calibration_versions installs other tables in the same process."""
    saved_cal = calibration._CAL
    saved_warned = set(calibration._WARNED_KEYS)
    calibration._CAL = copy.deepcopy(RAW)
    calibration._WARNED_KEYS.clear()
    try:
        yield
    finally:
        calibration._CAL = saved_cal
        calibration._WARNED_KEYS.clear()
        calibration._WARNED_KEYS.update(saved_warned)


# --------------------------------------------------------------------------------------------------
# Scalar getters: key name and float coercion.
# --------------------------------------------------------------------------------------------------


def test_get_comm_scale_reads_comm_scale_key():
    result = calibration.get_comm_scale()
    assert result == pytest.approx(RAW["comm_scale"])
    assert isinstance(result, float)


def test_get_training_overhead_s_reads_its_key():
    result = calibration.get_training_overhead_s()
    assert result == pytest.approx(RAW["training_overhead_s"])
    assert isinstance(result, float)


def test_get_mfu_batch_scale_returns_alpha_then_beta():
    # alpha != beta in the file (0.0341 vs 0.8147), so a swapped tuple fails.
    alpha, beta = calibration.get_mfu_batch_scale()
    s = RAW["mfu_batch_scale"]
    assert alpha == pytest.approx(s["alpha"])
    assert beta == pytest.approx(s["beta"])
    assert isinstance(alpha, float) and isinstance(beta, float)


# --------------------------------------------------------------------------------------------------
# Table getters: every fitted key maps to its JSON value (mapping over the whole fitted catalog).
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("gpu_name", sorted(RAW["mfu_multiplier"]))
def test_get_mfu_multiplier_known_keys_map_to_json(gpu_name):
    result = calibration.get_mfu_multiplier(gpu_name)
    assert result == pytest.approx(RAW["mfu_multiplier"][gpu_name])
    assert isinstance(result, float)


@pytest.mark.parametrize("method", sorted(RAW["method_scale"]))
def test_get_method_scale_known_keys_map_to_json(method):
    result = calibration.get_method_scale(method)
    assert result == pytest.approx(RAW["method_scale"][method])
    assert isinstance(result, float)


@pytest.mark.parametrize("model_name", sorted(RAW["model_scale"]))
def test_get_model_scale_known_keys_map_to_json(model_name):
    result = calibration.get_model_scale(model_name)
    assert result == pytest.approx(RAW["model_scale"][model_name])
    assert isinstance(result, float)


@pytest.mark.parametrize("sample_key", sorted(RAW["interaction_scale"]))
def test_get_interaction_scale_known_cells_map_to_json(sample_key):
    # Also checks the key format "model|method|gpu|num_gpus": a wrong separator or field order
    # builds a key absent from the table and returns the 1.0 default.
    model_name, method, gpu_name, num_gpus = sample_key.split("|")
    result = calibration.get_interaction_scale(model_name, method, gpu_name, int(num_gpus))
    assert result == pytest.approx(RAW["interaction_scale"][sample_key])
    assert isinstance(result, float)


# --------------------------------------------------------------------------------------------------
# Uncovered-key fallback: 1.0 with a warning (one branch per table).
# --------------------------------------------------------------------------------------------------


def test_get_mfu_multiplier_unknown_gpu_falls_back_to_one_with_warning():
    # an uncalibrated GPU leaves the prediction unchanged (1.0) and warns
    with pytest.warns(UserWarning, match="NVIDIA-DOES-NOT-EXIST"):
        result = calibration.get_mfu_multiplier("NVIDIA-DOES-NOT-EXIST")
    assert result == 1.0


def test_get_method_scale_unknown_method_falls_back_to_one_with_warning():
    with pytest.warns(UserWarning, match="definitely-not-a-method"):
        result = calibration.get_method_scale("definitely-not-a-method")
    assert result == 1.0


def test_get_model_scale_uncalibrated_model_falls_back_to_one_with_warning():
    assert "totally-uncalibrated-model" not in RAW["model_scale"]
    with pytest.warns(UserWarning, match="totally-uncalibrated-model"):
        result = calibration.get_model_scale("totally-uncalibrated-model")
    assert result == 1.0


def test_fallback_warning_fires_at_most_once_per_key():
    # _WARNED_KEYS dedupes: one warning across repeated lookups of one key
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibration.get_mfu_multiplier("one-shot-gpu")
        calibration.get_mfu_multiplier("one-shot-gpu")
    assert sum("one-shot-gpu" in str(w.message) for w in caught) == 1


# --------------------------------------------------------------------------------------------------
# multi_gpu_correction: 1.0 at <=1 GPU, exact hits, nearest-neighbour snap, tie-break.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_gpus", [1, 0, -5])
def test_get_multi_gpu_correction_single_or_below_is_unity_and_silent(num_gpus):
    # <=1 GPU has no all-reduce cost, so the divisor is 1.0 and nothing warns.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a snap warning would raise
        result = calibration.get_multi_gpu_correction(num_gpus)
    assert result == 1.0


@pytest.mark.parametrize("num_gpus", sorted(int(k) for k in RAW["multi_gpu_correction"]["by_num_gpus"]))
def test_get_multi_gpu_correction_exact_keys_return_value_without_snapping(num_gpus):
    # Each fitted count, including 16 and 64, returns its own value without a warning.
    expected = RAW["multi_gpu_correction"]["by_num_gpus"][str(num_gpus)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = calibration.get_multi_gpu_correction(num_gpus)
    assert result == pytest.approx(expected)
    assert isinstance(result, float)


def test_get_multi_gpu_correction_absent_count_snaps_to_nearest():
    # 5 is absent; |5-4|=1 < |5-8|=3 -> snaps to fitted count 4 (no interpolation), and warns.
    table = RAW["multi_gpu_correction"]["by_num_gpus"]
    assert "5" not in table
    with pytest.warns(UserWarning, match="num_gpus=5"):
        result = calibration.get_multi_gpu_correction(5)
    assert result == pytest.approx(table["4"])
    # no interpolation between the 4 and 8 values
    assert result != pytest.approx((table["4"] + table["8"]) / 2)


def test_get_multi_gpu_correction_above_max_snaps_to_largest_fitted_count():
    table = RAW["multi_gpu_correction"]["by_num_gpus"]
    max_key = max(int(k) for k in table)
    with pytest.warns(UserWarning, match=f"num_gpus={max_key + 1000}"):
        result = calibration.get_multi_gpu_correction(max_key + 1000)
    assert result == pytest.approx(table[str(max_key)])


def test_get_multi_gpu_correction_equidistant_tie_breaks_to_smaller_count():
    # 6 is equidistant from 4 and 8; min() over the ascending-insertion dict returns the first
    # minimum, the smaller count 4.
    table = RAW["multi_gpu_correction"]["by_num_gpus"]
    assert "6" not in table and {"4", "8"} <= set(table)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = calibration.get_multi_gpu_correction(6)
    assert result == pytest.approx(table["4"])
    assert result != pytest.approx(table["8"])


# --------------------------------------------------------------------------------------------------
# interaction_scale: absent cell -> neutral 1.0, no warning, no nearest-neighbour snapping.
# --------------------------------------------------------------------------------------------------


def test_get_interaction_scale_keys_str_enum_arguments_by_their_value():
    # f"{member}" of a (str, Enum) renders "Name.MEMBER", which built a key absent from the table.
    class Model(str, Enum):
        M = "granite-3.1-2b"

    class Meth(str, Enum):
        LORA = "lora"

    class Gpu(str, Enum):
        A100 = "NVIDIA-A100-SXM4-80GB"

    key = "granite-3.1-2b|lora|NVIDIA-A100-SXM4-80GB|2"
    assert key in RAW["interaction_scale"]
    assert RAW["interaction_scale"][key] != 1.0
    got = calibration.get_interaction_scale(Model.M, Meth.LORA, Gpu.A100, 2)
    assert got == pytest.approx(RAW["interaction_scale"][key])


def test_get_interaction_scale_absent_cell_defaults_to_one_silently():
    # interaction_scale does not snap: a fitted row queried at an unfitted GPU count returns 1.0
    # without a warning.
    key = "mistral-7b-v0.1|full|NVIDIA-A100-SXM4-80GB|16"
    assert key not in RAW["interaction_scale"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a snap-style warning would raise
        result = calibration.get_interaction_scale("mistral-7b-v0.1", "full", "NVIDIA-A100-SXM4-80GB", 16)
    assert result == 1.0


# --------------------------------------------------------------------------------------------------
# Loader tolerance and the lazy default load.
# --------------------------------------------------------------------------------------------------


def test_loader_tolerates_unknown_top_level_keys():
    # Extra top-level keys do not break the getters (no strict schema check).
    extra = {**RAW, "an_unknown_future_key": {"nested": 1}, "another": 42}
    saved = calibration._CAL
    try:
        calibration._CAL = extra
        assert calibration.get_comm_scale() == pytest.approx(RAW["comm_scale"])
    finally:
        calibration._CAL = saved


def test_active_calibration_lazy_loads_shipped_default(monkeypatch):
    # With no table installed and no $KAVIER_CALIBRATION, the first getter call loads the shipped
    # root calibration.json.
    monkeypatch.delenv("KAVIER_CALIBRATION", raising=False)
    calibration._CAL = None
    assert calibration.get_comm_scale() == pytest.approx(RAW["comm_scale"])
