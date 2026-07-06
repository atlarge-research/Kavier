"""Guards for the dev-only calibration regenerator (kavier.sdk.training.calibration.engine) and the
import-light accessor it feeds.

Two layers are exercised here:
  * engine.py pure helpers (mdape / _dumps / _neutral_base / _is_physical) -- fast, but engine.py
    imports scipy/sklearn/numpy/pandas at module top, so they gate on the [calibration] extra.
  * the cold-cache accessor path in __init__.py -- dependency-free.
  * the FROM-SCRATCH determinism guards (regenerate == committed file byte-for-byte) -- slow
    (~3 min each) and need the (unvendored) profiling traces; they skip on a clean checkout.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


def _engine():
    """Import engine.py, skipping when the heavy fit deps are absent (clean checkout)."""
    pytest.importorskip("scipy")  # engine.py imports scipy/sklearn at module top
    pytest.importorskip("sklearn")
    from kavier.sdk.training.calibration import engine

    return engine


# ==================================== mdape ======================================
def test_mdape_returns_the_median_not_the_mean_of_abs_pct_errors():
    mdape = _engine().mdape
    # measured all 100; pred 110/120/150 -> abs pct errors 10/20/50. median=20, mean=26.667.
    # A mean-based impl would return 26.667, so this pins median specifically.
    got = mdape(np.array([100.0, 100.0, 100.0]), np.array([110.0, 120.0, 150.0]))
    assert got == pytest.approx(20.0)
    assert got != pytest.approx((10.0 + 20.0 + 50.0) / 3.0)  # not the mean


def test_mdape_ignores_rows_with_nonpositive_measured():
    mdape = _engine().mdape
    # row0 has measured=0 and must be dropped; only row1 counts: |150-100|/100 = 50%.
    # Dropping the mask would fold row0's huge error in and change the median.
    assert mdape(np.array([0.0, 100.0]), np.array([999.0, 150.0])) == pytest.approx(50.0)


def test_mdape_is_nan_when_no_row_has_positive_measured():
    mdape = _engine().mdape
    # No measured > 0 -> undefined -> NaN (not 0.0, which would masquerade as a perfect fit).
    assert np.isnan(mdape(np.array([0.0, -1.0]), np.array([5.0, 5.0])))


# ==================================== _dumps =====================================
def test_dumps_uses_indent_two_and_a_single_trailing_newline():
    _dumps = _engine()._dumps
    # Byte-for-byte regeneration hinges on this exact serialisation. Hand-written expected string
    # (json indent=2, newline-terminated); a switch to indent=4 or a dropped "\n" fails this.
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

    # Every multiplicative correction neutralised to 1.0 (comm_scale was 1.23).
    assert nb["comm_scale"] == 1.0
    assert set(nb["mfu_multiplier"].values()) == {1.0}
    assert set(nb["method_scale"].values()) == {1.0}
    assert set(nb["multi_gpu_correction"]["by_num_gpus"].values()) == {1.0}
    # model_scale restricted to the requested set (m2 dropped), order preserved, all reset to 1.0.
    assert list(nb["model_scale"]) == ["m1", "m3"]
    assert set(nb["model_scale"].values()) == {1.0}
    # interaction_scale emptied.
    assert nb["interaction_scale"] == {}
    # Raw-physics constants + schema/version are NOT fits -> carried through unchanged.
    assert nb["mfu_batch_scale"] == {"alpha": 0.1, "beta": 0.2}
    assert nb["training_overhead_s"] == 0.5
    assert (nb["schema_version"], nb["version"]) == ("x", "y")
    # Does not mutate the caller's reference.
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
    """Regression (0.5.0 anchor move): with ``_CAL`` reset the first accessor call must re-run
    ``importlib.resources.files()`` on the calibration anchor and resolve the shipped package -- not
    raise "not a package". Independent oracle: the value must equal calibration.json's comm_scale
    read straight off disk, which also proves it resolved the RIGHT file, not merely some file."""
    import os
    from pathlib import Path

    import kavier.sdk.training.calibration as cal

    if os.environ.get("KAVIER_CALIBRATION"):
        pytest.skip("KAVIER_CALIBRATION overrides which file the default accessor loads")

    # Independent oracle: read the shipped calibration.json straight off disk. Resolve it from the
    # import-light package (NOT engine.py, which imports scipy/sklearn -- absent on a clean checkout);
    # this is the same file as engine.CAL_PATH: <calibration package>/calibration.json.
    cal_path = Path(cal.__file__).parent / "calibration.json"
    expected = json.loads(cal_path.read_text(encoding="utf-8"))["comm_scale"]
    saved = cal._CAL
    try:
        cal._CAL = None  # force importlib.resources.files() + a fresh default load
        assert cal.get_comm_scale() == pytest.approx(expected)  # must not raise; must match the file
    finally:
        cal._CAL = saved


# ========================= from-scratch determinism (slow) =========================
def _used_values(cal: dict) -> dict:
    """The calibration with the catalog-dependent, UNUSED >8-GPU multi_gpu_correction stripped out.

    The >8-GPU mgc (16/32/64/128) is a global median fit over EVERY catalog model's big-GPU rows in the
    raw multi-node trace, so it legitimately shifts whenever the catalog gains models (the 23 added for
    the separate calibration_all use-case did exactly this) -- while the shipped *thesis* calibration
    files (4-LLM validation, 6-LLM exploration) stay frozen. The recommender restricts to <=8 GPU, so
    these values are never consumed. Stripping them lets the determinism guard still pin, byte-for-byte,
    every value that IS used (comm_scale, mfu_multiplier, method/model scale, interaction, mgc 2/4/8),
    without coupling the frozen thesis files to catalog growth. See docs/content/known-weaknesses.md."""
    out = json.loads(json.dumps(cal))  # cheap deep copy
    by = out["multi_gpu_correction"]["by_num_gpus"]
    out["multi_gpu_correction"]["by_num_gpus"] = {k: v for k, v in by.items() if int(k) <= 8}
    out["multi_gpu_correction"].pop("_note", None)  # the note narrates the >8 fit
    return out


def test_regenerate_reproduces_committed_calibration():
    """Mirrors ``engine --check``: regenerating the calibration FROM SCRATCH (carrying nothing from any
    prior table) reproduces every USED value of the committed calibration.json exactly (the unused,
    catalog-dependent >8-GPU mgc is excluded -- see _used_values). Independent oracle = the committed
    file. A non-deterministic fit (unseeded split/optimiser) or any perturbed <=8 formula breaks this.
    Slow (~3 min) + needs the [calibration] extra and the unvendored traces."""
    engine = _engine()
    from kavier.sdk.training.calibration.engine import CAL_PATH, TRACE_ARCHIVE

    if not (TRACE_ARCHIVE / engine.PROFILING_CSV).exists():
        pytest.skip("internal profiling trace not vendored")

    ref_text = CAL_PATH.read_text(encoding="utf-8")
    regenerated = engine.regenerate(json.loads(ref_text), TRACE_ARCHIVE)
    # Used values pinned exactly; the >8 mgc is compared structurally (same GPU-count keys) only.
    assert _used_values(regenerated) == _used_values(json.loads(ref_text))
    assert set(regenerated["multi_gpu_correction"]["by_num_gpus"]) == set(
        json.loads(ref_text)["multi_gpu_correction"]["by_num_gpus"]
    )


@pytest.mark.parametrize("model_set", ["6", "4"])
def test_regenerate_reproduces_versions_file_for_each_model_set(model_set):
    """Same determinism guard per model-set: rebuilding a set from scratch (the profiling trace
    filtered to it) reproduces every USED value of its versions/ file exactly (the unused,
    catalog-dependent >8-GPU mgc excluded -- see _used_values). Slow; skips without the extra/traces."""
    engine = _engine()
    from kavier.sdk.training.calibration.engine import CAL_PATH, TRACE_ARCHIVE

    if not (TRACE_ARCHIVE / engine.PROFILING_CSV).exists():
        pytest.skip("internal profiling trace not vendored")

    reference = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    regenerated = engine.regenerate(reference, TRACE_ARCHIVE, models=engine.MODEL_SETS[model_set])
    target = json.loads(engine.VERSION_FILES[model_set].read_text(encoding="utf-8"))
    assert _used_values(regenerated) == _used_values(target)
    assert set(regenerated["multi_gpu_correction"]["by_num_gpus"]) == set(target["multi_gpu_correction"]["by_num_gpus"])
