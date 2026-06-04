"""Tests for the Kavier empirical calibration layer.

Covers ``kavier_training.core.calibration`` getters and the on-disk
``data/calibration.json`` they read. These are pure lookups over a fitted
JSON table; the tests pin down the contract (valid JSON / expected keys,
correct value per known key, documented fallback behaviours, and strict
KeyError on uncalibrated entries) without touching production code.

Run with:
    cd kavier
    PYTHONPATH=src ../.venv/bin/python -m pytest \
        src/tests/test_training/test_calibration.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kavier_training.core import calibration

# Load the raw JSON independently of the module so we can cross-check that the
# getters return exactly what is on disk (rather than re-hardcoding constants).
_CAL_PATH = Path(calibration.__file__).resolve().parent.parent / "data" / "calibration.json"
with _CAL_PATH.open(encoding="utf-8") as _f:
    RAW = json.load(_f)


# --------------------------------------------------------------------------- #
# calibration.json structure
# --------------------------------------------------------------------------- #
def test_calibration_json_is_valid_json_and_loads():
    # Loading happens at import time inside the module; re-load here to assert
    # the file on disk is well-formed JSON yielding a dict.
    with _CAL_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_calibration_json_has_schema_version_provenance():
    # Provenance: the on-disk format carries an integer schema_version, and the
    # loader records the version it targets.
    assert RAW["schema_version"] == 1
    assert calibration.SCHEMA_VERSION == 1


def test_loader_tolerates_unknown_top_level_keys():
    # Descriptive/extra top-level keys (schema_version, version, _note, and any
    # future addition) must not break loading or the getters.
    extra = {**RAW, "an_unknown_future_key": {"nested": 1}, "another": 42}
    saved = calibration._CAL
    try:
        calibration._CAL = extra
        # Getters still work, reading only the keys they need.
        assert calibration.get_comm_scale() == pytest.approx(RAW["comm_scale"])
        assert calibration.get_method_scale(next(iter(RAW["method_scale"]))) == pytest.approx(
            next(iter(RAW["method_scale"].values()))
        )
    finally:
        calibration._CAL = saved


@pytest.mark.parametrize(
    "key, expected_type",
    [
        ("comm_scale", (int, float)),
        ("training_overhead_s", (int, float)),
        ("mfu_multiplier", dict),
        ("multi_gpu_correction", dict),
        ("method_scale", dict),
        ("model_scale", dict),
        ("interaction_scale", dict),
    ],
)
def test_calibration_json_has_expected_keys(key, expected_type):
    assert key in RAW
    assert isinstance(RAW[key], expected_type)


def test_multi_gpu_correction_has_by_num_gpus_subtable():
    assert "by_num_gpus" in RAW["multi_gpu_correction"]
    assert isinstance(RAW["multi_gpu_correction"]["by_num_gpus"], dict)
    assert len(RAW["multi_gpu_correction"]["by_num_gpus"]) > 0


# --------------------------------------------------------------------------- #
# Scalar getters
# --------------------------------------------------------------------------- #
def test_get_comm_scale_matches_json():
    assert calibration.get_comm_scale() == pytest.approx(RAW["comm_scale"])
    assert isinstance(calibration.get_comm_scale(), float)


def test_get_training_overhead_s_matches_json():
    assert calibration.get_training_overhead_s() == pytest.approx(RAW["training_overhead_s"])
    assert isinstance(calibration.get_training_overhead_s(), float)


# --------------------------------------------------------------------------- #
# get_mfu_multiplier
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gpu_name", sorted(RAW["mfu_multiplier"]))
def test_get_mfu_multiplier_known_keys(gpu_name):
    result = calibration.get_mfu_multiplier(gpu_name)
    assert result == pytest.approx(RAW["mfu_multiplier"][gpu_name])
    assert isinstance(result, float)


def test_get_mfu_multiplier_unknown_gpu_raises_keyerror():
    with pytest.raises(KeyError):
        calibration.get_mfu_multiplier("NVIDIA-DOES-NOT-EXIST")


# --------------------------------------------------------------------------- #
# get_method_scale
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", sorted(RAW["method_scale"]))
def test_get_method_scale_known_keys(method):
    result = calibration.get_method_scale(method)
    assert result == pytest.approx(RAW["method_scale"][method])
    assert isinstance(result, float)


def test_get_method_scale_unknown_method_raises_keyerror():
    with pytest.raises(KeyError):
        calibration.get_method_scale("definitely-not-a-method")


# --------------------------------------------------------------------------- #
# get_model_scale
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_name", sorted(RAW["model_scale"]))
def test_get_model_scale_known_keys(model_name):
    result = calibration.get_model_scale(model_name)
    assert result == pytest.approx(RAW["model_scale"][model_name])
    assert isinstance(result, float)


def test_get_model_scale_uncalibrated_model_raises_keyerror():
    # Strict: an uncalibrated model must NOT silently fall back to a default.
    assert "totally-uncalibrated-model" not in RAW["model_scale"]
    with pytest.raises(KeyError):
        calibration.get_model_scale("totally-uncalibrated-model")


# --------------------------------------------------------------------------- #
# get_multi_gpu_correction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_gpus", [0, 1, -5])
def test_get_multi_gpu_correction_single_or_below_is_unity(num_gpus):
    # <= 1 GPU means no multi-GPU communication penalty.
    assert calibration.get_multi_gpu_correction(num_gpus) == 1.0


@pytest.mark.parametrize("num_gpus", sorted(int(k) for k in RAW["multi_gpu_correction"]["by_num_gpus"]))
def test_get_multi_gpu_correction_exact_table_keys(num_gpus):
    expected = RAW["multi_gpu_correction"]["by_num_gpus"][str(num_gpus)]
    result = calibration.get_multi_gpu_correction(num_gpus)
    assert result == pytest.approx(expected)
    assert isinstance(result, float)


def test_get_multi_gpu_correction_nearest_neighbour_for_absent_count():
    # Table has 2,4,8,32,128 but not 5. 5 is closer to 4 (dist 1) than 8 (dist 3),
    # so the getter should snap to the value for 4 GPUs.
    table = RAW["multi_gpu_correction"]["by_num_gpus"]
    assert "5" not in table
    assert calibration.get_multi_gpu_correction(5) == pytest.approx(table["4"])


def test_get_multi_gpu_correction_nearest_neighbour_above_max():
    # A count larger than every table key snaps to the largest key (128 here).
    table = RAW["multi_gpu_correction"]["by_num_gpus"]
    max_key = max(int(k) for k in table)
    assert calibration.get_multi_gpu_correction(max_key + 1000) == pytest.approx(table[str(max_key)])


def test_get_multi_gpu_correction_nearest_neighbour_midpoint_uses_smaller():
    # 6 is equidistant from 4 and 8 (dist 2 each). min() over the comprehension
    # keeps the first minimal element, which for the sorted dict iteration is the
    # smaller key (4). This pins down the documented tie behaviour.
    table = RAW["multi_gpu_correction"]["by_num_gpus"]
    assert "6" not in table
    assert {"4", "8"} <= set(table)
    assert calibration.get_multi_gpu_correction(6) == pytest.approx(table["4"])


# --------------------------------------------------------------------------- #
# get_interaction_scale
# --------------------------------------------------------------------------- #
def test_get_interaction_scale_known_combo_matches_json():
    # Pick a real key from the fitted table and verify it round-trips.
    sample_key = next(iter(RAW["interaction_scale"]))
    model_name, method, gpu_name, num_gpus = sample_key.split("|")
    result = calibration.get_interaction_scale(model_name, method, gpu_name, int(num_gpus))
    assert result == pytest.approx(RAW["interaction_scale"][sample_key])
    assert isinstance(result, float)


@pytest.mark.parametrize("sample_key", list(RAW["interaction_scale"]))
def test_get_interaction_scale_all_known_keys_match_json(sample_key):
    model_name, method, gpu_name, num_gpus = sample_key.split("|")
    result = calibration.get_interaction_scale(model_name, method, gpu_name, int(num_gpus))
    assert result == pytest.approx(RAW["interaction_scale"][sample_key])


def test_get_interaction_scale_absent_combo_defaults_to_one():
    # Unseen (model, method, gpu, gpu-count) combination -> neutral 1.0 so it
    # leaves the prediction unchanged.
    key = "no-such-model|full|NVIDIA-A100-SXM4-80GB|1"
    assert key not in RAW["interaction_scale"]
    assert calibration.get_interaction_scale("no-such-model", "full", "NVIDIA-A100-SXM4-80GB", 1) == 1.0


def test_get_interaction_scale_known_model_unknown_gpu_count_defaults_to_one():
    # Same model/method/gpu as a fitted row but a gpu-count not in the table:
    # no nearest-neighbour fallback here, it must default to 1.0.
    key = "mistral-7b-v0.1|full|NVIDIA-A100-SXM4-80GB|16"
    assert key not in RAW["interaction_scale"]
    assert calibration.get_interaction_scale("mistral-7b-v0.1", "full", "NVIDIA-A100-SXM4-80GB", 16) == 1.0
