"""Behaviour tests for the kavier.sdk.training.calibration getters over the shipped calibration.json.

The getters are thin accessors, so the meaningful behaviours are: (1) each getter reads the CORRECT
JSON key and coerces to float, (2) uncovered keys fall back to a NEUTRAL 1.0 with a one-time warning,
(3) multi_gpu_correction clamps to unity for <=1 GPU and snaps absent counts to the nearest fitted
count (no interpolation), and (4) interaction_scale defaults absent cells to 1.0 without warning or
nearest-neighbour snapping. The raw JSON is loaded independently from disk so the key-mapping asserts
are cross-checked against the data file, never against the getter's own output.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import pytest

from kavier.sdk.training import calibration

# Independently load the shipped calibration.json straight from disk. Used both as the table the
# getters read (via the autouse fixture) and as the oracle for key-mapping asserts.
_CAL_PATH = Path(calibration.__file__).resolve().parent / "calibration.json"
with _CAL_PATH.open(encoding="utf-8") as _f:
    RAW = json.load(_f)


@pytest.fixture(autouse=True)
def _pin_shipped_calibration():
    """Isolate every test from cross-file _CAL swaps (test_calibration_versions installs other
    tables in the same process) and from stale one-time-warning state. Pin the live table to a fresh
    copy of the shipped JSON and clear the warned-key set, then restore afterwards."""
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
# Scalar getters: correct key name + float coercion.
# --------------------------------------------------------------------------------------------------


def test_get_comm_scale_reads_comm_scale_key():
    # Falsifies: getter returns a constant, or reads a different key than "comm_scale".
    result = calibration.get_comm_scale()
    assert result == pytest.approx(RAW["comm_scale"])
    assert isinstance(result, float)


def test_get_training_overhead_s_reads_its_key():
    # Falsifies: getter returns a constant, or reads a different key than "training_overhead_s".
    result = calibration.get_training_overhead_s()
    assert result == pytest.approx(RAW["training_overhead_s"])
    assert isinstance(result, float)


def test_get_mfu_batch_scale_returns_alpha_then_beta():
    # (alpha, beta) ORDER matters: alpha!=beta in the file (0.0341 vs 0.8147), so a swapped return
    # would flip which value each name holds and fail. Falsifies swapped tuple / wrong subkeys.
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
    # Falsifies: getter hardcodes one value, or looks up the wrong table.
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
    # Also exercises the key CONSTRUCTION "model|method|gpu|num_gpus": a wrong separator or field
    # order would build a key absent from the table and return the 1.0 default -> mismatch.
    model_name, method, gpu_name, num_gpus = sample_key.split("|")
    result = calibration.get_interaction_scale(model_name, method, gpu_name, int(num_gpus))
    assert result == pytest.approx(RAW["interaction_scale"][sample_key])
    assert isinstance(result, float)


# --------------------------------------------------------------------------------------------------
# Uncovered-key fallback: neutral 1.0 with a warning (three independent branches, one per table).
# --------------------------------------------------------------------------------------------------


def test_get_mfu_multiplier_unknown_gpu_falls_back_to_one_with_warning():
    # Oracle: an uncalibrated GPU must not perturb the prediction -> neutral 1.0, and must warn.
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
    # Oracle: _WARNED_KEYS dedupes -> exactly one warning across repeated lookups of one bad key.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibration.get_mfu_multiplier("one-shot-gpu")
        calibration.get_mfu_multiplier("one-shot-gpu")
    assert sum("one-shot-gpu" in str(w.message) for w in caught) == 1


# --------------------------------------------------------------------------------------------------
# multi_gpu_correction: unity clamp, exact hits, nearest-neighbour snap, tie-break.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_gpus", [1, 0, -5])
def test_get_multi_gpu_correction_single_or_below_is_unity_and_silent(num_gpus):
    # Analytic oracle: <=1 GPU has no ring-all-reduce comm penalty -> divisor exactly 1.0, no warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any snap warning would raise and fail
        result = calibration.get_multi_gpu_correction(num_gpus)
    assert result == 1.0


@pytest.mark.parametrize("num_gpus", sorted(int(k) for k in RAW["multi_gpu_correction"]["by_num_gpus"]))
def test_get_multi_gpu_correction_exact_keys_return_value_without_snapping(num_gpus):
    # Every fitted count returns its own value and MUST NOT warn (guards the regression where 16/64
    # were snapped instead of stored). Falsifies: exact branch warns, or returns a neighbour's value.
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
    # Cross-check it did NOT interpolate to the midpoint of the 4 and 8 values.
    assert result != pytest.approx((table["4"] + table["8"]) / 2)


def test_get_multi_gpu_correction_above_max_snaps_to_largest_fitted_count():
    table = RAW["multi_gpu_correction"]["by_num_gpus"]
    max_key = max(int(k) for k in table)
    with pytest.warns(UserWarning, match=f"num_gpus={max_key + 1000}"):
        result = calibration.get_multi_gpu_correction(max_key + 1000)
    assert result == pytest.approx(table[str(max_key)])


def test_get_multi_gpu_correction_equidistant_tie_breaks_to_smaller_count():
    # 6 is equidistant from 4 and 8 (dist 2 each); min() over the ascending-insertion dict returns
    # the first minimum -> the smaller count 4. Falsifies: tie resolved to 8, or to their average.
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


def test_get_interaction_scale_absent_cell_defaults_to_one_silently():
    # Unlike multi_gpu_correction, interaction_scale has NO snapping: a fitted row queried at an
    # unfitted gpu-count returns the neutral 1.0 default (prediction unchanged) and does not warn.
    key = "mistral-7b-v0.1|full|NVIDIA-A100-SXM4-80GB|16"
    assert key not in RAW["interaction_scale"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a snap-style warning would raise
        result = calibration.get_interaction_scale("mistral-7b-v0.1", "full", "NVIDIA-A100-SXM4-80GB", 16)
    assert result == 1.0


# --------------------------------------------------------------------------------------------------
# Loader robustness and the lazy default-load path.
# --------------------------------------------------------------------------------------------------


def test_loader_tolerates_unknown_top_level_keys():
    # Extra/descriptive top-level keys must not break the getters (guards against strict-schema
    # rejection). Falsifies: a getter that validates the table shape and raises on unknown keys.
    extra = {**RAW, "an_unknown_future_key": {"nested": 1}, "another": 42}
    saved = calibration._CAL
    try:
        calibration._CAL = extra
        assert calibration.get_comm_scale() == pytest.approx(RAW["comm_scale"])
    finally:
        calibration._CAL = saved


def test_active_calibration_lazy_loads_shipped_default(monkeypatch):
    # With no table installed and no $KAVIER_CALIBRATION override, the first getter call must lazily
    # load the shipped root calibration.json. Falsifies: lazy default reads the wrong file / no file.
    monkeypatch.delenv("KAVIER_CALIBRATION", raising=False)
    calibration._CAL = None
    assert calibration.get_comm_scale() == pytest.approx(RAW["comm_scale"])
