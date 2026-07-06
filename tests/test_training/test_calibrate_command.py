"""Guards for the ``kavier calibrate`` command (CLI in kavier.cli.calibrate, SDK fit in
kavier.sdk.training.calibration.engine.calibrate).

``kavier calibrate`` exposes the from-scratch calibration fit as a data-driven command: it runs the
SAME two-tier Powell + interaction_scale recipe as ``regenerate``, parameterized on the given file --
BUT, unlike regenerate, with NO <=8-GPU cap, so a dataset's >8-GPU rows join the main joint fit
directly. Because of that it no longer reproduces calibration.json byte-for-byte (regenerate() does;
its byte-identity guard lives in test_engine_regen.py). The correctness anchor here is instead that
calibrate() IS ``_fit_calibration`` re-parameterized on the same any-GPU data (a still-falsifiable
oracle), plus structural / plausibility checks and the robustness contracts (missing-column ValueError,
thin-dataset warning, per-regime report). Fit-running tests need the [calibration] extra (scipy/sklearn)
and so skip on a clean checkout, like test_engine_regen.py; the registration / dispatch / missing-extra
tests are dependency-free.
"""

from __future__ import annotations

import json
import sys

import pytest


def _fit_deps_or_skip():
    """Import the engine, skipping when the heavy fit deps (scipy/sklearn) are absent."""
    pytest.importorskip("scipy")  # engine.py imports scipy/sklearn at module top
    pytest.importorskip("sklearn")
    import pandas as pd  # noqa: F401  (ensures pandas present too)

    from kavier.sdk.training.calibration import engine

    return engine


def _synthetic_trace(models, gpu="NVIDIA-A100-SXM4-80GB", *, totals=((2, 1), (8, 1), (8, 2)), batches=(4, 8)):
    """A small but STRUCTURALLY-COMPLETE profiling frame over ``models``: real catalog model/GPU names
    (so simulate_training_step resolves), a spread of batch sizes, and a spread of total-GPU counts that
    INCLUDES a >8 count (8 GPUs x 2 nodes = 16) -- exercising calibrate's uncapped >8-in-main-fit path.
    ``totals`` are (number_gpus, number_nodes) pairs. dataset_tokens_per_second is a deterministic ramp."""
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


# ============================ correctness anchor (replaces byte-identity) ============================
def test_calibrate_is_fit_calibration_reparameterized_on_uncapped_data():
    """calibrate(df, models) must equal a DIRECT ``_fit_calibration`` call on the SAME any-GPU rows --
    i.e. calibrate is literally that recipe re-parameterized, applying NO GPU-count cap.

    Independent oracle: ``_fit_calibration(reference, rows, None, models)`` computed here from the SAME
    reference (the shipped template, unchanged because both models are already in it) and the SAME
    uncapped filtered rows -- not a hard-coded table. Falsification: if calibrate silently re-imposed a
    <=8 cap, used a different reference/model list/split, or diverged from the recipe, the >8 (16-GPU)
    rows would drop or the fit would differ and the two dumps would stop matching. Plus structural /
    plausibility checks (expected keys present; comm_scale + every mfu_multiplier finite and inside the
    [_SCALE_LO,_SCALE_HI] band the fit is supposed to guarantee; model_scale finite/positive)."""
    engine = _fit_deps_or_skip()
    from kavier.sdk.training.calibration.engine import CAL_PATH

    models = ["granite-3-8b", "mistral-7b-v0.1"]  # both already in the template -> no model_scale seeding
    df = _synthetic_trace(models)

    # Oracle: reproduce calibrate's own (reference, rows) prep, then call the recipe directly.
    reference = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    rows = engine._filter_valid_rows(df, models, max_total_gpus=None)
    assert (rows["total"] > 8).any(), "fixture must carry >8-GPU rows to test the uncapped path"
    expected, _metrics = engine._fit_calibration(reference, rows, None, models, log=lambda _m: None)

    actual = engine.calibrate(df, models=models)

    # Same recipe, same data -> byte-identical serialization (the strongest still-true equality).
    assert engine._dumps(actual) == engine._dumps(expected)

    # Structural: the full shipped schema is present.
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

    # Plausibility band: the accept-guard is supposed to keep these scales physical.
    import math

    lo, hi = engine._SCALE_LO, engine._SCALE_HI
    assert math.isfinite(actual["comm_scale"]) and lo <= actual["comm_scale"] <= hi
    for g, v in actual["mfu_multiplier"].items():
        assert math.isfinite(v) and lo <= v <= hi, f"mfu_multiplier[{g}]={v} outside [{lo},{hi}]"
    for m in models:
        assert math.isfinite(actual["model_scale"][m]) and actual["model_scale"][m] > 0


def test_calibrate_missing_required_column_raises_valueerror_naming_it():
    """A missing REQUIRED column is the ONE hard failure: a clear ValueError naming exactly the absent
    column(s), not a downstream KeyError.

    Oracle: the documented required-column contract. We drop two known-required columns and assert both
    are named (and a still-present required column is NOT named). Falsification: a KeyError/other
    exception, a message omitting a dropped column, or a message that falsely names a present column."""
    engine = _fit_deps_or_skip()

    df = _synthetic_trace(["granite-3-8b"]).drop(columns=["batch_size", "method"])
    with pytest.raises(ValueError) as exc:
        engine.calibrate(df, models=["granite-3-8b"])
    msg = str(exc.value)
    assert "batch_size" in msg and "method" in msg  # both dropped columns named
    assert "dataset_tokens_per_second" not in msg.split("required:")[0]  # a present col isn't flagged as missing


def test_calibrate_thin_dataset_warns_with_specifics_and_still_returns_a_table():
    """A thin-but-structurally-valid dataset (one model, 3 rows, constant batch, single GPU) must STILL
    fit and return a valid table, while emitting the headline suitability warning that names EXACTLY the
    violated properties for this tiny hand-built input.

    Oracle: the three suitability thresholds hand-applied to the fixture -- 3 rows < 30 (rows/model);
    1 distinct batch size < 2 (batch spread); 1 distinct GPU-count < 2 (GPU-count spread). Falsification:
    no warning, a warning that omits any of the three violations or the offending model/GPU, or a crash
    instead of a returned table."""
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
    # Names the too-few-rows violation for the specific model (n=3 hand-derived).
    assert "granite-3-8b (n=3)" in msg and str(engine.MIN_ROWS_PER_MODEL) in msg
    # Names the constant-batch (model, GPU) cell.
    assert "granite-3-8b/NVIDIA-A100-SXM4-80GB" in msg and "batch size" in msg
    # Names the single-GPU-count violation.
    assert "1 distinct total-GPU count" in msg and "[1]" in msg

    # Still a valid, complete table despite the thin data (never crashes).
    assert out["model_scale"]["granite-3-8b"] > 0
    for key in ("comm_scale", "mfu_multiplier", "multi_gpu_correction", "interaction_scale"):
        assert key in out


def test_calibrate_suitable_dataset_emits_no_headline_warning():
    """The mirror of the thin-dataset test: a dataset meeting all thresholds must NOT emit the headline
    suitability warning (so the warning is a real signal, not always-on noise).

    Oracle: the same thresholds, now all satisfied -- >=30 rows/model, >=2 batch sizes, >=2 GPU-counts.
    Falsification: the 'Tuning may have produced poor results' warning firing on clean data."""
    engine = _fit_deps_or_skip()

    # >=30 rows for the single model across many batch sizes and 4 GPU-counts (incl. a >8 count).
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
    """The CLI's stderr summary must include the held-out test MdAPE broken down BY MODEL and BY
    GPU-COUNT, each on the same seed-42 test split.

    Oracle: the per-regime reporting contract -- for every model and every total-GPU count that lands in
    the test split there is a labelled line. We reconstruct that split independently (the same
    _filter_valid_rows + train_val_test_split the CLI uses) and assert each expected model name and each
    expected GPU-count label appears under its section. Falsification: a missing section, or a
    model/GPU-count present in the test split but absent from the printed breakdown."""
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

    # Independently derive which models / GPU-counts fall in the seed-42 test split.
    valid = engine._filter_valid_rows(df, models, max_total_gpus=None)
    _, _, test = engine.train_val_test_split(valid)
    for m in sorted(test["model_name"].astype(str).unique()):
        assert m in err, f"model {m} in test split but missing from the per-model breakdown"
    for c in sorted(int(t) for t in test["total"].unique()):
        assert f"{c:>4} GPU" in err, f"GPU-count {c} in test split but missing from the breakdown"


@pytest.mark.parametrize(
    ("gpu", "method", "novel_key", "table"),
    [
        # A100-80GB is a REAL catalog GPU (so simulate_training_step resolves it) but is NOT one of the
        # template's 4 mfu_multiplier keys; before the fix this bare-KeyError-crashed the Powell layout.
        ("A100-80GB", "full", "A100-80GB", "mfu_multiplier"),
        # fsdp is outside the template's method_scale {full, gptq-lora, lora}; same crash class.
        ("NVIDIA-A100-SXM4-80GB", "fsdp", "fsdp", "method_scale"),
    ],
)
def test_calibrate_fits_novel_gpu_or_method_instead_of_keyerror_crashing(gpu, method, novel_key, table):
    """A GPU or training method PRESENT in the data but ABSENT from the shipped template must be
    calibrated from its own rows, not crash -- calibrate() seeds a neutral 1.0 prior for novel GPUs and
    methods just as it does for novel models. (Regression: adversarial robustness review found that a
    non-template GPU/method hit a bare KeyError in the layout lookup, violating calibrate's documented
    'a missing REQUIRED column is the only hard failure' contract.)

    Oracle: that contract -- calibrate returns a well-formed table in which the novel key is present with
    a finite, in-band scale. Falsification: a KeyError (or any exception), or the novel key absent from
    the returned table (a silent drop)."""
    engine = _fit_deps_or_skip()
    import math
    import warnings

    df = _synthetic_trace(["granite-3-8b"], gpu=gpu)
    df["method"] = method

    with warnings.catch_warnings():  # the thin-data suitability warning is orthogonal here
        warnings.simplefilter("ignore")
        cal = engine.calibrate(df, models=["granite-3-8b"])  # must not raise

    assert novel_key in cal[table], f"{novel_key!r} was dropped from {table} instead of being fit"
    lo, hi = engine._SCALE_LO, engine._SCALE_HI
    v = cal[table][novel_key]
    assert math.isfinite(v) and lo <= v <= hi, f"{table}[{novel_key}]={v} outside [{lo},{hi}]"


# ============================== dependency-free command wiring ==============================
def test_calibrate_missing_extra_prints_install_hint_and_exits_nonzero(monkeypatch, capsys):
    """Without scipy/sklearn the command must fail FAST with the documented install hint, not a raw
    ImportError traceback. We simulate the absent extra by poisoning the engine module in sys.modules
    (import of a None entry raises ImportError), so the test holds even with the extra installed.

    Oracle: the command's contract -- non-zero exit + the exact ``uv sync --extra calibration`` hint on
    stderr. Falsification: swallowing the ImportError, exiting 0, a traceback, or a different message."""
    from kavier.cli import calibrate as cli

    monkeypatch.setitem(sys.modules, "kavier.sdk.training.calibration.engine", None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["nonexistent.csv"])  # arg parses fine; the engine import fails before any file read

    assert exc.value.code not in (0, None)  # non-zero: a usable exit status for a caller/subprocess
    err = capsys.readouterr().err
    assert "uv sync --extra calibration" in err  # the actionable, copy-pasteable fix
    assert "[calibration] extra" in err


def test_calibrate_registered_and_dispatches_argv(monkeypatch):
    """`calibrate` must be wired into the unified dispatcher AND forward its trailing argv untouched.

    Oracle: the dispatch contract -- _COMMANDS['calibrate'] routes to kavier.cli.calibrate.main with
    the exact remaining argv. Falsification: command absent from the table, or a handler that drops /
    rewrites argv. (Stubs main, so no fit runs -- dependency-free.)"""
    from kavier.cli import calibrate
    from kavier.cli.main import _COMMANDS

    assert "calibrate" in _COMMANDS
    help_text, handler = _COMMANDS["calibrate"]
    assert help_text  # a one-line help string is present

    seen: dict[str, object] = {}
    monkeypatch.setattr(calibrate, "main", lambda argv=None: seen.__setitem__("argv", argv))
    handler(["trace.csv", "--models", "granite-3-8b", "--output", "out.json"])
    assert seen["argv"] == ["trace.csv", "--models", "granite-3-8b", "--output", "out.json"]


def test_calibrate_help_exits_zero_and_documents_flags(capsys):
    """`kavier calibrate --help` is a success (exit 0) and advertises the input arg + every flag.

    Oracle: argparse's --help contract + this command's documented surface (positional input,
    --output, --models). Falsification: non-zero exit or a flag missing from the help body."""
    from kavier.cli import calibrate as cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "kavier calibrate" in out
    for token in ("input", "--output", "--models"):
        assert token in out
