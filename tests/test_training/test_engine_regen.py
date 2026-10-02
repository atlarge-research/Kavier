"""Tests for the dev-only calibration regenerator (kavier.sdk.training.calibration.engine) and its accessor.

* engine.py helpers (mdape, _dumps, _neutral_base, _is_physical): fast, but engine.py imports
  scipy/sklearn/numpy/pandas at module top, so they need the [calibration] extra.
* the cold-cache accessor path in __init__.py: stdlib only.
* from-scratch determinism (regenerate equals the committed file): ~3 min each, needs the
  unvendored profiling traces, skips on a clean checkout.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


def _engine():
    """Import engine.py, or skip when the fit dependencies are missing."""
    pytest.importorskip("scipy")  # engine.py imports scipy/sklearn at module top
    pytest.importorskip("sklearn")
    from kavier.sdk.training.calibration import engine

    return engine


# ==================================== mdape ======================================
def test_mdape_returns_the_median_not_the_mean_of_abs_pct_errors():
    mdape = _engine().mdape
    # measured all 100; pred 110/120/150 -> abs pct errors 10/20/50. median=20, mean=26.667.
    got = mdape(np.array([100.0, 100.0, 100.0]), np.array([110.0, 120.0, 150.0]))
    assert got == pytest.approx(20.0)
    assert got != pytest.approx((10.0 + 20.0 + 50.0) / 3.0)


def test_mdape_ignores_rows_with_nonpositive_measured():
    mdape = _engine().mdape
    # row0 has measured=0 and is dropped; only row1 counts: |150-100|/100 = 50%.
    # Without the mask row0's large error would change the median.
    assert mdape(np.array([0.0, 100.0]), np.array([999.0, 150.0])) == pytest.approx(50.0)


def test_mdape_is_nan_when_no_row_has_positive_measured():
    mdape = _engine().mdape
    # No measured > 0: undefined, so NaN. 0.0 would read as a perfect fit.
    assert np.isnan(mdape(np.array([0.0, -1.0]), np.array([5.0, 5.0])))


# ==================================== _dumps =====================================
def test_dumps_uses_indent_two_and_a_single_trailing_newline():
    _dumps = _engine()._dumps
    # Byte-for-byte regeneration depends on this serialization: json indent=2, one trailing newline.
    expected = '{\n  "a": 1,\n  "b": [\n    1,\n    2\n  ]\n}\n'
    assert _dumps({"a": 1, "b": [1, 2]}) == expected


# ================================= _neutral_base =================================
def test_neutral_base_resets_every_correction_to_one_and_filters_models():
    _neutral_base = _engine()._neutral_base
    reference = {
        "schema_version": "x",
        "version": "y",
        "comm_scale": 1.23,
        "training_overhead_s": 0.5,
        "mfu_batch_scale": {"alpha": 0.1, "beta": 0.2},
        "mfu_multiplier": {"G1": 1.5, "G2": 0.7},
        "multi_gpu_correction": {"by_num_gpus": {"2": 1.3, "4": 1.7}},
        "method_scale": {"full": 0.9, "lora": 1.4},
        "model_scale": {"m1": 1.1, "m2": 0.8, "m3": 1.2},
        "interaction_scale": {"m1|full|G1|1": 1.05},
    }
    nb = _neutral_base(reference, ["m1", "m3"])

    # every multiplicative correction reset to 1.0
    assert nb["comm_scale"] == 1.0
    assert set(nb["mfu_multiplier"].values()) == {1.0}
    assert set(nb["method_scale"].values()) == {1.0}
    assert set(nb["multi_gpu_correction"]["by_num_gpus"].values()) == {1.0}
    # model_scale restricted to the requested set (m2 dropped), order preserved, all reset to 1.0.
    assert list(nb["model_scale"]) == ["m1", "m3"]
    assert set(nb["model_scale"].values()) == {1.0}
    assert nb["interaction_scale"] == {}
    # raw-physics constants and schema/version are not fitted and pass through
    assert nb["mfu_batch_scale"] == {"alpha": 0.1, "beta": 0.2}
    assert nb["training_overhead_s"] == 0.5
    assert (nb["schema_version"], nb["version"]) == ("x", "y")
    # the caller's reference is not mutated
    assert reference["comm_scale"] == 1.23


# ================================= _is_physical ==================================
@pytest.mark.parametrize(
    ("comm_scale", "mfu", "expected"),
    [
        (0.5, 2.0, True),  # both endpoints of the [0.5, 2.0] band are inclusive
        (0.49, 1.0, False),  # comm_scale just below the floor
        (2.01, 1.0, False),  # comm_scale just above the ceiling
        (1.0, 2.01, False),  # a single mfu_multiplier above the ceiling fails the AND
        (1.0, 0.49, False),  # a single mfu_multiplier below the floor fails the AND
    ],
)
def test_is_physical_accepts_only_the_closed_band(comm_scale, mfu, expected):
    _is_physical = _engine()._is_physical
    cal = {"comm_scale": comm_scale, "mfu_multiplier": {"g": mfu}}
    assert _is_physical(cal) is expected


# ============================== cold-cache accessor ==============================
def test_cold_cache_accessor_resolves_package_and_reads_shipped_comm_scale():
    """With ``_CAL`` reset, the first accessor call resolves the package through ``importlib.resources.files()``.

    The returned comm_scale must equal the one in calibration.json read from disk."""
    import os
    from pathlib import Path

    import kavier.sdk.training.calibration as cal

    if os.environ.get("KAVIER_CALIBRATION"):
        pytest.skip("KAVIER_CALIBRATION overrides which file the default accessor loads")

    # Path taken from the stdlib-only package; engine.py needs scipy/sklearn. Same file as engine.CAL_PATH.
    cal_path = Path(cal.__file__).parent / "calibration.json"
    expected = json.loads(cal_path.read_text(encoding="utf-8"))["comm_scale"]
    saved = cal._CAL
    try:
        cal._CAL = None  # force importlib.resources.files() and a fresh default load
        assert cal.get_comm_scale() == pytest.approx(expected)
    finally:
        cal._CAL = saved


# ========================= from-scratch determinism (slow) =========================
def _used_values(cal: dict) -> dict:
    """Return the calibration without its >8-GPU multi_gpu_correction entries.

    The >8-GPU mgc (16/32/64/128) is a global median over every catalog model's large-GPU rows in the
    raw multi-node trace, so it changes when models are added to the catalog, while the thesis calibration
    files (4-LLM validation, 6-LLM exploration) stay frozen. The recommender uses only <=8 GPUs and never
    reads these values. All used values (comm_scale, mfu_multiplier, method/model scale, interaction,
    mgc 2/4/8) are still compared exactly."""
    out = json.loads(json.dumps(cal))  # deep copy
    by = out["multi_gpu_correction"]["by_num_gpus"]
    out["multi_gpu_correction"]["by_num_gpus"] = {k: v for k, v in by.items() if int(k) <= 8}
    out["multi_gpu_correction"].pop("_note", None)  # the note describes the >8 fit
    return out


def test_regenerate_reproduces_committed_calibration():
    """Regenerating from scratch reproduces every used value of the committed calibration.json.

    Same check as ``engine --check``; the >8-GPU mgc is excluded (see _used_values). An unseeded split
    or optimizer, or a changed <=8 formula, breaks it. ~3 min; needs the [calibration] extra and the
    unvendored traces."""
    engine = _engine()
    from kavier.sdk.training.calibration.engine import CAL_PATH, TRACE_ARCHIVE

    if not (TRACE_ARCHIVE / engine.PROFILING_CSV).exists():
        pytest.skip("internal profiling trace not vendored")

    ref_text = CAL_PATH.read_text(encoding="utf-8")
    regenerated = engine.regenerate(json.loads(ref_text), TRACE_ARCHIVE)
    # used values compared exactly; the >8 mgc on its GPU-count keys only
    assert _used_values(regenerated) == _used_values(json.loads(ref_text))
    assert set(regenerated["multi_gpu_correction"]["by_num_gpus"]) == set(
        json.loads(ref_text)["multi_gpu_correction"]["by_num_gpus"]
    )


@pytest.mark.parametrize("model_set", ["6", "4"])
def test_regenerate_reproduces_versions_file_for_each_model_set(model_set):
    """Rebuilding each model set from the filtered profiling trace reproduces every used value of its versions/ file.

    The >8-GPU mgc is excluded (see _used_values). Slow; skips without the extra or the traces."""
    engine = _engine()
    from kavier.sdk.training.calibration.engine import CAL_PATH, TRACE_ARCHIVE

    if not (TRACE_ARCHIVE / engine.PROFILING_CSV).exists():
        pytest.skip("internal profiling trace not vendored")

    reference = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    regenerated = engine.regenerate(reference, TRACE_ARCHIVE, models=engine.MODEL_SETS[model_set])
    target = json.loads(engine.VERSION_FILES[model_set].read_text(encoding="utf-8"))
    assert _used_values(regenerated) == _used_values(target)
    assert set(regenerated["multi_gpu_correction"]["by_num_gpus"]) == set(target["multi_gpu_correction"]["by_num_gpus"])


# ================================ CLI write guard ================================
def _run_engine_cli(monkeypatch, engine, argv, written):
    """Run engine.main() with ``argv``; regenerate returns the shipped table and _write_set calls go to ``written``."""
    reference = json.loads(engine.CAL_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(engine, "regenerate", lambda *a, **k: reference)
    monkeypatch.setattr(engine, "_write_set", lambda model_set, text: written.append(model_set))
    monkeypatch.setattr("sys.argv", ["engine", *argv])
    engine.main()


def test_cli_refuses_write_with_a_custom_model_list(monkeypatch):
    """--write with --models used to write the custom fit over the 6-model file and calibration.json."""
    engine = _engine()
    written: list[str] = []
    with pytest.raises(SystemExit) as exc:
        _run_engine_cli(monkeypatch, engine, ["--models", "granite-3-8b", "--write"], written)
    assert exc.value.code == 2
    assert written == []


def test_cli_writes_a_custom_model_fit_only_to_out(monkeypatch, tmp_path):
    engine = _engine()
    out = tmp_path / "custom.json"
    written: list[str] = []
    _run_engine_cli(monkeypatch, engine, ["--models", "granite-3-8b", "--out", str(out)], written)
    assert written == []
    assert out.read_text(encoding="utf-8") == engine.CAL_PATH.read_text(encoding="utf-8")
