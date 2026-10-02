"""Tests for ``kavier calibrate`` (CLI in kavier.cli.calibrate, fit in kavier.sdk.training.calibration.engine).

The command runs the two-tier Powell + interaction_scale fit of ``regenerate`` on the given file, but
without the <=8-GPU cap, so >8-GPU rows join the main joint fit. It therefore does not reproduce
calibration.json byte-for-byte; that check is in test_engine_regen.py. Here the reference is a direct
``_fit_calibration`` call on the same uncapped rows, plus structural and plausibility checks, the
missing-column ValueError, the thin-dataset warning and the per-regime report. Fit tests need the
[calibration] extra (scipy, sklearn) and skip without it; the registration, dispatch and missing-extra
tests have no extra dependencies.
"""

from __future__ import annotations

import json
import sys

import pytest


def _fit_deps_or_skip():
    """Import the engine, or skip when scipy or sklearn is missing."""
    pytest.importorskip("scipy")  # engine.py imports scipy/sklearn at module top
    pytest.importorskip("sklearn")
    import pandas as pd  # noqa: F401  (pandas must be importable too)

    from kavier.sdk.training.calibration import engine

    return engine


def _synthetic_trace(models, gpu="NVIDIA-A100-SXM4-80GB", *, totals=((2, 1), (8, 1), (8, 2)), batches=(4, 8)):
    """Return a small profiling frame over ``models`` with catalog model and GPU names.

    Covers several batch sizes and total-GPU counts, including 16 (8 GPUs x 2 nodes) for the uncapped
    >8-GPU path. ``totals`` are (number_gpus, number_nodes) pairs; dataset_tokens_per_second is a
    deterministic ramp.
    """
    import pandas as pd

    rows = []
    tps = 900.0
    for m in models:
        for g, n in totals:
            for b in batches:
                tps += 25.0
                rows.append(
                    {
                        "model_name": m,
                        "gpu_model": gpu,
                        "method": "full",
                        "tokens_per_sample": 512,
                        "batch_size": b,
                        "number_gpus": g,
                        "number_nodes": n,
                        "is_valid": 1,
                        "dataset_tokens_per_second": tps * g * n,
                    }
                )
    return pd.DataFrame(rows)


# ===================================== fit equivalence ======================================
def test_calibrate_is_fit_calibration_reparameterized_on_uncapped_data():
    """calibrate(df, models) equals a direct ``_fit_calibration`` call on the same rows, with no GPU-count cap.

    The reference is ``_fit_calibration(reference, rows, None, models)`` on the shipped template (both
    models are already in it) and the same uncapped rows. A <=8 cap, a different reference, model list
    or split, or a changed recipe drops the 16-GPU rows or changes the fit. Also checks the expected
    keys, that comm_scale and every mfu_multiplier are finite and inside [_SCALE_LO, _SCALE_HI], and
    that model_scale is finite and positive."""
    engine = _fit_deps_or_skip()
    from kavier.sdk.training.calibration.engine import CAL_PATH

    models = ["granite-3-8b", "mistral-7b-v0.1"]  # both already in the template -> no model_scale seeding
    df = _synthetic_trace(models)

    # Same (reference, rows) preparation as calibrate, then the recipe called directly.
    reference = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    rows = engine._filter_valid_rows(df, models, max_total_gpus=None)
    assert (rows["total"] > 8).any(), "fixture must carry >8-GPU rows to test the uncapped path"
    expected, _metrics = engine._fit_calibration(reference, rows, None, models, log=lambda _m: None)

    actual = engine.calibrate(df, models=models)

    # same recipe and data -> identical serialization
    assert engine._dumps(actual) == engine._dumps(expected)

    for key in (
        "comm_scale",
        "mfu_multiplier",
        "multi_gpu_correction",
        "method_scale",
        "model_scale",
        "interaction_scale",
        "schema_version",
        "version",
    ):
        assert key in actual, f"missing calibration key {key!r}"

    # The fit's accept guard keeps these scales inside the physical band.
    import math

    lo, hi = engine._SCALE_LO, engine._SCALE_HI
    assert math.isfinite(actual["comm_scale"]) and lo <= actual["comm_scale"] <= hi
    for g, v in actual["mfu_multiplier"].items():
        assert math.isfinite(v) and lo <= v <= hi, f"mfu_multiplier[{g}]={v} outside [{lo},{hi}]"
    for m in models:
        assert math.isfinite(actual["model_scale"][m]) and actual["model_scale"][m] > 0


def test_calibrate_missing_required_column_raises_valueerror_naming_it():
    """A missing required column raises a ValueError that names the absent columns.

    Two required columns are dropped; both must be named and a present required column must not be."""
    engine = _fit_deps_or_skip()

    df = _synthetic_trace(["granite-3-8b"]).drop(columns=["batch_size", "method"])
    with pytest.raises(ValueError) as exc:
        engine.calibrate(df, models=["granite-3-8b"])
    msg = str(exc.value)
    assert "batch_size" in msg and "method" in msg
    assert "dataset_tokens_per_second" not in msg.split("required:")[0]  # a present column is not listed as missing


def test_calibrate_thin_dataset_warns_with_specifics_and_still_returns_a_table():
    """A thin but well-formed dataset still fits and returns a table, with one suitability warning.

    For this input (one model, 3 rows, constant batch, single GPU) the warning names all three violations:
    3 rows < 30 per model, 1 distinct batch size < 2, 1 distinct GPU count < 2."""
    engine = _fit_deps_or_skip()
    import pandas as pd

    df = pd.DataFrame(
        {
            "model_name": ["granite-3-8b"] * 3,
            "gpu_model": ["NVIDIA-A100-SXM4-80GB"] * 3,
            "method": ["full"] * 3,
            "tokens_per_sample": [512, 512, 512],
            "batch_size": [4, 4, 4],  # constant batch -> 1 distinct
            "number_gpus": [1, 1, 1],  # single GPU -> 1 distinct total-GPU count
            "number_nodes": [1, 1, 1],
            "is_valid": [1, 1, 1],
            "dataset_tokens_per_second": [1000.0, 1100.0, 1050.0],
        }
    )

    with pytest.warns(UserWarning) as record:
        out = engine.calibrate(df, models=["granite-3-8b"])

    suitability = [str(w.message) for w in record if "Tuning may have produced poor results" in str(w.message)]
    assert len(suitability) == 1, "exactly one headline suitability warning expected"
    msg = suitability[0]
    # too few rows for the model (n=3)
    assert "granite-3-8b (n=3)" in msg and str(engine.MIN_ROWS_PER_MODEL) in msg
    # constant batch size in the (model, GPU) cell
    assert "granite-3-8b/NVIDIA-A100-SXM4-80GB" in msg and "batch size" in msg
    # single total-GPU count
    assert "1 distinct total-GPU count" in msg and "[1]" in msg

    # the table is still complete
    assert out["model_scale"]["granite-3-8b"] > 0
    for key in ("comm_scale", "mfu_multiplier", "multi_gpu_correction", "interaction_scale"):
        assert key in out


def test_calibrate_suitable_dataset_emits_no_headline_warning():
    """A dataset meeting every threshold emits no suitability warning.

    Thresholds: >=30 rows per model, >=2 batch sizes, >=2 total-GPU counts."""
    engine = _fit_deps_or_skip()

    # 32 rows: 8 batch sizes x 4 GPU counts, one of them >8
    df = _synthetic_trace(
        ["granite-3-8b"],
        totals=((2, 1), (4, 1), (8, 1), (8, 2)),
        batches=(1, 2, 4, 8, 16, 32, 64, 128),
    )
    df = df.reset_index(drop=True)
    assert len(df) >= engine.MIN_ROWS_PER_MODEL

    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        engine.calibrate(df, models=["granite-3-8b"])
    assert not [w for w in record if "Tuning may have produced poor results" in str(w.message)]


def test_calibrate_cli_reports_per_regime_mdape_by_model_and_gpu_count(tmp_path, capsys):
    """The CLI's stderr summary reports held-out test MdAPE by model and by GPU count.

    The seed-42 test split is rebuilt with the _filter_valid_rows and train_val_test_split calls the CLI
    uses; every model and total-GPU count in it must have a labelled line."""
    engine = _fit_deps_or_skip()
    from kavier.cli import calibrate as cli

    models = ["granite-3-8b", "mistral-7b-v0.1"]
    df = _synthetic_trace(models)
    csv = tmp_path / "trace.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "cal.json"

    cli.main([str(csv), "--output", str(out), "--models", ",".join(models)])
    err = capsys.readouterr().err

    assert "test MdAPE by model:" in err
    assert "test MdAPE by GPU-count" in err

    # models and GPU counts in the seed-42 test split
    valid = engine._filter_valid_rows(df, models, max_total_gpus=None)
    _, _, test = engine.train_val_test_split(valid)
    for m in sorted(test["model_name"].astype(str).unique()):
        assert m in err, f"model {m} in test split but missing from the per-model breakdown"
    for c in sorted(int(t) for t in test["total"].unique()):
        assert f"{c:>4} GPU" in err, f"GPU-count {c} in test split but missing from the breakdown"


@pytest.mark.parametrize(
    ("gpu", "method", "novel_key", "table"),
    [
        # A100-80GB is in the catalog but not among the template's 4 mfu_multiplier keys.
        ("A100-80GB", "full", "A100-80GB", "mfu_multiplier"),
        # fsdp is outside the template's method_scale {full, gptq-lora, lora}.
        ("NVIDIA-A100-SXM4-80GB", "fsdp", "fsdp", "method_scale"),
    ],
)
def test_calibrate_fits_novel_gpu_or_method_instead_of_keyerror_crashing(gpu, method, novel_key, table):
    """A GPU or method in the data but absent from the shipped template is fitted from its own rows.

    calibrate() seeds a neutral 1.0 prior for new GPUs and methods, as it does for new models. The
    returned table must contain the new key with a finite scale inside the band. A missing required
    column is the only input that may raise."""
    engine = _fit_deps_or_skip()
    import math
    import warnings

    df = _synthetic_trace(["granite-3-8b"], gpu=gpu)
    df["method"] = method

    with warnings.catch_warnings():  # the thin-data warning is not under test here
        warnings.simplefilter("ignore")
        cal = engine.calibrate(df, models=["granite-3-8b"])

    assert novel_key in cal[table], f"{novel_key!r} was dropped from {table} instead of being fit"
    lo, hi = engine._SCALE_LO, engine._SCALE_HI
    v = cal[table][novel_key]
    assert math.isfinite(v) and lo <= v <= hi, f"{table}[{novel_key}]={v} outside [{lo},{hi}]"


# ==================================== helpers of the fit ====================================
def test_mgc_gaps_below_the_lowest_fitted_count_interpolate_from_one_gpu():
    """Counts below the lowest fitted count are log2-interpolated between mgc(1) = 1 and that count.

    Holding them flat at mgc(16) = 2.5 made 2 GPUs slower than 1. With the anchor, mgc(k) = 2.5 ** (log2(k) / 4)
    for k = 2, 4, 8, rounded to 2 dp like the other interpolated counts."""
    engine = _fit_deps_or_skip()
    fitted = {"2": 1.0, "4": 1.0, "8": 1.0, "16": 2.5, "32": 3.0, "64": 1.0, "128": 1.0}
    out = engine._mgc_with_interpolated_gaps(fitted, {16, 32})
    assert out["2"] == round(2.5**0.25, 2)  # 1.26
    assert out["4"] == round(2.5**0.5, 2)  # 1.58
    assert out["8"] == round(2.5**0.75, 2)  # 1.99
    assert (out["16"], out["32"]) == (2.5, 3.0)  # fitted counts keep their value
    assert (out["64"], out["128"]) == (3.0, 3.0)  # above the top fitted count: held flat


def test_mgc_with_nothing_fitted_stays_neutral():
    engine = _fit_deps_or_skip()
    out = engine._mgc_with_interpolated_gaps({"2": 1.0, "4": 1.0, "16": 1.0}, set())
    assert out == {"2": 1.0, "4": 1.0, "16": 1.0}


def test_filter_valid_rows_skips_invalid_rows_with_missing_gpu_counts():
    """An invalid row with a NaN GPU count is dropped; the int cast used to run on it and raise."""
    engine = _fit_deps_or_skip()
    import numpy as np
    import pandas as pd

    df = _synthetic_trace(["granite-3-8b"])
    bad = df.iloc[[0]].copy()
    bad["number_gpus"] = np.nan
    bad["is_valid"] = 0
    with_bad = pd.concat([df, bad], ignore_index=True)

    got = engine._filter_valid_rows(with_bad, None, max_total_gpus=8)
    expected = engine._filter_valid_rows(df, None, max_total_gpus=8)
    assert list(got.index) == list(expected.index)
    assert got["total"].tolist() == expected["total"].tolist()


# ============================== dependency-free command wiring ==============================
def test_calibrate_missing_extra_prints_install_hint_and_exits_nonzero(monkeypatch, capsys):
    """Without scipy/sklearn the command exits non-zero and prints the ``uv sync --extra calibration`` hint.

    The engine entry in sys.modules is set to None, so its import raises ImportError even when the extra
    is installed."""
    from kavier.cli import calibrate as cli

    monkeypatch.setitem(sys.modules, "kavier.sdk.training.calibration.engine", None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["nonexistent.csv"])  # the engine import fails before the file is read

    assert exc.value.code not in (0, None)
    err = capsys.readouterr().err
    assert "uv sync --extra calibration" in err
    assert "[calibration] extra" in err


def test_calibrate_registered_and_dispatches_argv(monkeypatch):
    """`calibrate` is registered in the dispatcher and receives the remaining argv unchanged.

    calibrate.main is stubbed, so no fit runs."""
    from kavier.cli import calibrate
    from kavier.cli.main import _COMMANDS, main

    assert "calibrate" in _COMMANDS
    help_text, _module = _COMMANDS["calibrate"]
    assert help_text

    seen: dict[str, object] = {}
    monkeypatch.setattr(calibrate, "main", lambda argv=None: seen.__setitem__("argv", argv))
    main(["calibrate", "trace.csv", "--models", "granite-3-8b", "--output", "out.json"])
    assert seen["argv"] == ["trace.csv", "--models", "granite-3-8b", "--output", "out.json"]


def test_calibrate_help_exits_zero_and_documents_flags(capsys):
    """`kavier calibrate --help` exits 0 and lists the input argument, --output and --models."""
    from kavier.cli import calibrate as cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "kavier calibrate" in out
    for token in ("input", "--output", "--models"):
        assert token in out
