#!/usr/bin/env python
"""The calibration engine: ONE file with all the dev-only fitting logic for calibration.json.

Kavier predicts training speed from physics. Calibration is a small set of "correction factors"
-- learned from real measured runs -- that nudge those predictions closer to reality (physics-only
is ~16% off; calibrated is ~10%). This module rebuilds them FROM SCRATCH from the data, carrying
NOTHING from any previous calibration. The lean runtime accessor lives in __init__.py; the fitted
tables live in calibration.json (the default) + versions/; everything that *produces* them is here.

The from-scratch table is fit from two inputs only:
  1. raw (uncalibrated) kavier   -- the physics, with every correction reset to a neutral 1.0
  2. the measured profiling runs -- trace-archive/profiling-dataset/profiling_trace.csv

How it works -- two layers (see ``regenerate``):
  - Tier-1 (global scales) = Powell minimization (derivative-free, not gradient) in log-space,
    minimizing median-APE + an L2 pull toward neutral, lambda chosen on validation.
  - Tier-2 (per-config) = median of measured/predicted ratios.
  (>8-GPU mgc 32/128: same median-ratio on the multi-node trace; 16/64: log2 geomean. The
   held-out 15% test split is never fit -- it only reports accuracy.)

Two model-sets are fit from the same recipe (the profiling trace is filtered to the set):
  - 4-model (dense-4: mistral-7b-v0.1, granite-3.3-8b, granite-3-8b, llama3.2-3b) -> versions/calibration_4model.json
  - 6-model (dense-4 + granite-3.1-2b + granite-3.1-8b-instruct)                  -> versions/calibration_6model.json
The 6-model fit is the default shipped table (calibration.json); the two files are byte-identical.

Usage (ENG = kavier.sdk.training.calibration.engine):
  python -m ENG --check                # rebuild BOTH sets; confirm each == its versions/ file
  python -m ENG --write                # rebuild --model-set (default 6) + overwrite its file(s)
  python -m ENG --model-set 4 --write  # rebuild + write only the 4-model file
  python -m ENG --snapshot             # rebuild + save a timestamped copy to diff

The measured-runs files are internal and not shipped here; point at them with --profiling-data / --raw-trace.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY
from kavier.sdk.library.llm import LLM_SPEC_LIBRARY
from kavier.sdk.training.core.engine import simulate_training_step

# ================================ paths & constants ================================
# engine.py sits at src/kavier/sdk/training/calibration/, so:
# parents: [4]=src  [5]=repo root  [6]=workspace (holds sibling trace-archive)
_HERE = Path(__file__).resolve()
SRC = _HERE.parents[4]
REPO_ROOT = _HERE.parents[5]
WORKSPACE = _HERE.parents[6]
TRACE_ARCHIVE = WORKSPACE / "trace-archive" / "profiling-dataset"
CAL_PATH = _HERE.parent / "calibration.json"

# ONE merged profiling export = curated dense-4 rows (across GPU types) + the controlled-benchmark
# granite rows. This is the only input the <=8-GPU Tier-1/Tier-2 fit reads.
PROFILING_CSV = "profiling_trace.csv"
# Raw multi-node trace, read only to fit the >8-GPU multi-GPU correction (32/128 GPUs).
RAW_MULTINODE_CSV = "raw_trace.csv"

# Physical-plausibility band for the Tier-1 selection. At <=8-GPU single-node, communication is tiny
# next to compute, so comm_scale -- and, under weak regularization, a GPU's mfu_multiplier -- is
# nearly UNCONSTRAINED by the data: an unregularized fit (lambda=0) drives comm_scale to a
# noise-fitting ~20x. Such a fit predicts the <=8 test split about as well, but it is not physical
# (a shipped "physics + small corrections" table with comm_scale=20 is wrong on its face) and it
# corrupts the >8-GPU mgc fit, which divides comm back out. We therefore only accept a Tier-1
# candidate whose comm_scale and every mfu_multiplier sit in this band, then pick the best-validation
# survivor. Regularizing a non-identifiable parameter toward its neutral prior is the correct
# behaviour; the honest held-out test accuracy is ~10% (comm_scale ~ 1.23).
_SCALE_LO, _SCALE_HI = 0.5, 2.0

# Selectable model-sets. The recipe is identical; only the profiling rows it fits on differ
# (the trace is filtered to the set). DENSE_4 is the exp1 head-to-head set; ALL_6 adds the two
# controlled-benchmark granite-3.1 models (exp4 / in-vitro). ALL_6 reproduces calibration.json.
DENSE_4 = ["mistral-7b-v0.1", "granite-3.3-8b", "granite-3-8b", "llama3.2-3b"]
ALL_6 = [*DENSE_4, "granite-3.1-2b", "granite-3.1-8b-instruct"]
MODEL_SETS: dict[str, list[str]] = {"4": DENSE_4, "6": ALL_6}

# The two from-scratch calibrations live here; calibration.json (root) == versions/calibration_6model.json.
VERSIONS_DIR = CAL_PATH.parent / "versions"
VERSION_FILES: dict[str, Path] = {
    "4": VERSIONS_DIR / "calibration_4model.json",
    "6": VERSIONS_DIR / "calibration_6model.json",
}

# Deterministic seed-42 70/15/15 split. NOTE: train_test_split's permutation depends on row COUNT +
# seed, but the rows it selects also depend on the input DataFrame's row ORDER. Byte-identical
# regeneration therefore assumes the canonical curated CSV (same rows, same order, same pre-filter);
# re-exporting it in a different order shifts which rows land in train/val and changes the fitted
# values. Do NOT sort the rows -- that would change the shipped numbers.
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.176  # 0.176 of the remaining 85% ~= 15% of total

# Columns every profiling trace MUST carry for the fit to be well-defined. A missing one is the only
# HARD failure the data-driven calibrate() raises (a clear ValueError naming the offenders); everything
# else (thin/narrow data) degrades to a warning + best-effort fit. Kept as a module constant so the
# check has a single source of truth shared by _filter_valid_rows and any caller that wants to pre-check.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "model_name",
    "gpu_model",
    "method",
    "tokens_per_sample",
    "batch_size",
    "number_gpus",
    "number_nodes",
    "is_valid",
    "dataset_tokens_per_second",
)

# "Suitable calibration dataset" thresholds -- used ONLY to emit the headline warning in calibrate();
# they never change the fit itself (a dataset that falls short is still fit, just flagged). See
# _suitability_report. MIN_ROWS_PER_MODEL: a model with fewer valid rows can't pin its scales well.
# MIN_DISTINCT_BATCH_SIZES: a single batch size per (model, GPU) leaves the batch/MFU shape unconstrained.
# MIN_DISTINCT_GPU_COUNTS: without >=2 distinct total-GPU counts the multi-GPU correction is unidentifiable.
MIN_ROWS_PER_MODEL = 30
MIN_DISTINCT_BATCH_SIZES = 2
MIN_DISTINCT_GPU_COUNTS = 2


@contextmanager
def calibration_override(cal_dict: dict[str, Any]) -> Iterator[None]:
    """Temporarily install ``cal_dict`` as the live calibration the engine reads, then restore.
    Swaps the module global kavier.sdk.training.calibration._CAL (the same contract Coastline uses)."""
    import kavier.sdk.training.calibration as cal

    saved = cal._CAL
    cal._CAL = cal_dict
    try:
        yield
    finally:
        cal._CAL = saved


# ==================================== metrics =====================================
def mdape(measured: np.ndarray, pred: np.ndarray) -> float:
    """Median absolute percentage error (%), over rows with ``measured > 0`` -- the metric the
    calibration fit minimizes / reports."""
    measured = np.asarray(measured, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = measured > 0
    if not m.any():
        return float("nan")
    return float(np.median(np.abs((pred[m] - measured[m]) / measured[m])) * 100.0)


# ===================================== split ======================================
def train_val_test_split(
    df: pd.DataFrame, *, test_size: float = TEST_SIZE, val_size: float = VAL_SIZE, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train, val, test) DataFrames (seed-42 70/15/15; see the SEED note)."""
    temp, test = train_test_split(df, test_size=test_size, random_state=seed)
    train, val = train_test_split(temp, test_size=val_size, random_state=seed)
    return train, val, test


# ================================ Tier-1 Powell fit ================================
# Regularized Powell joint refit, lambda chosen on validation. Refits the global scales (comm_scale,
# mfu_multiplier, present multi_gpu_correction keys, method_scale, model_scale) in log-space,
# minimizing train MdAPE + an L2 pull toward the prior; select_calibration picks lambda on
# validation. regenerate() drives this with base_cal = the neutral raw kavier (nothing carried).
_SHIPPED = json.loads(CAL_PATH.read_text())


def _powell_row_args(rows: pd.DataFrame) -> list[dict]:
    """Pre-extract per-row engine kwargs for fast repeated evaluation."""
    args = []
    for _, r in rows.iterrows():
        total = int(r["number_gpus"]) * int(r["number_nodes"])
        args.append(
            dict(
                model_name=str(r["model_name"]),
                gpu_model=str(r["gpu_model"]),
                tokens_per_sample=int(r["tokens_per_sample"]),
                batch_size=int(r["batch_size"]),
                method=str(r["method"]),
                num_gpus=total,
                num_nodes=int(r["number_nodes"]),
            )
        )
    return args


def _powell_predict(row_args: list[dict], grad_accum_steps: int, backward_factor: float, cal_dict: dict) -> np.ndarray:
    out = np.empty(len(row_args), dtype=np.float64)
    with calibration_override(cal_dict):
        for i, a in enumerate(row_args):
            out[i] = simulate_training_step(**a, grad_accum_steps=grad_accum_steps, backward_factor=backward_factor)[
                "tokens_per_second"
            ]
    return out


def _vary_layout(rows: pd.DataFrame, base_cal: dict) -> list[tuple]:
    """The (kind, key) scale entries the train rows exercise; everything else stays at base_cal.
    Only mgc keys already in the table are varied (absent totals keep the engine fallback)."""
    models = sorted(rows["model_name"].astype(str).unique())
    methods = sorted(rows["method"].astype(str).unique())
    gpus = sorted(rows["gpu_model"].astype(str).unique())
    totals = sorted({int(a) * int(b) for a, b in zip(rows["number_gpus"], rows["number_nodes"])})
    mgc_table = base_cal["multi_gpu_correction"]["by_num_gpus"]
    mgc_keys = [str(t) for t in totals if t > 1 and str(t) in mgc_table]
    layout: list[tuple] = []
    if any(t > 1 for t in totals):
        layout.append(("comm_scale", None))
    layout += [("mfu_multiplier", g) for g in gpus]
    layout += [("mgc", k) for k in mgc_keys]
    layout += [("method_scale", m) for m in methods]
    layout += [("model_scale", m) for m in models]
    return layout


def _get(cal_dict: dict, kind: str, key):
    if kind == "comm_scale":
        return cal_dict["comm_scale"]
    if kind == "mfu_multiplier":
        return cal_dict["mfu_multiplier"][key]
    if kind == "mgc":
        return cal_dict["multi_gpu_correction"]["by_num_gpus"][key]
    if kind == "method_scale":
        return cal_dict["method_scale"][key]
    if kind == "model_scale":
        return cal_dict["model_scale"][key]
    raise KeyError(kind)


def _apply(base_cal: dict, layout: list[tuple], values) -> dict:
    c = copy.deepcopy(base_cal)
    for (kind, key), v in zip(layout, values):
        v = float(v)
        if kind == "comm_scale":
            c["comm_scale"] = v
        elif kind == "mfu_multiplier":
            c["mfu_multiplier"][key] = v
        elif kind == "mgc":
            c["multi_gpu_correction"]["by_num_gpus"][key] = v
        elif kind == "method_scale":
            c["method_scale"][key] = v
        elif kind == "model_scale":
            c["model_scale"][key] = v
    return c


def _powell_fit(
    train_rows: pd.DataFrame,
    grad_accum_steps: int,
    backward_factor: float,
    base_cal: dict = _SHIPPED,
    maxiter: int = 60,
    lam: float = 0.0,
) -> dict:
    """Refit the exercised scales (Powell, log-space) from ``base_cal``, minimizing
    train MdAPE + ``lam`` * mean((log_scale - log_prior)**2)."""
    args = _powell_row_args(train_rows)
    y_true = pd.to_numeric(train_rows["dataset_tokens_per_second"], errors="coerce").to_numpy(np.float64)
    layout = _vary_layout(train_rows, base_cal)
    log_x0 = np.log(np.array([_get(base_cal, k, key) for (k, key) in layout], dtype=np.float64))

    def objective(log_x: np.ndarray) -> float:
        c = _apply(base_cal, layout, np.exp(log_x))
        md = mdape(y_true, _powell_predict(args, grad_accum_steps, backward_factor, c))
        if not np.isfinite(md):
            return 1e9
        penalty = lam * float(np.mean((log_x - log_x0) ** 2))
        return md + penalty

    res = minimize(objective, log_x0, method="Powell", options={"maxiter": maxiter, "maxfev": 8000})
    return _apply(base_cal, layout, np.exp(res.x))


def evaluate(rows: pd.DataFrame, grad_accum_steps: int, backward_factor: float, cal_dict: dict) -> float:
    """MdAPE (%) of ``cal_dict`` on ``rows``."""
    args = _powell_row_args(rows)
    y_true = pd.to_numeric(rows["dataset_tokens_per_second"], errors="coerce").to_numpy(np.float64)
    return mdape(y_true, _powell_predict(args, grad_accum_steps, backward_factor, cal_dict))


def select_calibration(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    grad_accum_steps: int,
    backward_factor: float,
    base_cal: dict = _SHIPPED,
    lambdas: tuple = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0),
    maxiter: int = 60,
    accept=None,
) -> tuple[dict, dict]:
    """Fit at each lambda on TRAIN; also consider base_cal (no refit). Pick the candidate with
    the best VALIDATION MdAPE. If ``accept`` is given, only candidates for which ``accept(cal)`` is
    True are eligible -- used to reject a physically-degenerate fit (e.g. a non-identifiable
    comm_scale that runs away under weak regularization). Returns (calibration, {choice, val_mdape})."""
    candidates = [("none(v2)", copy.deepcopy(base_cal))]
    for lam in lambdas:
        candidates.append(
            (f"lam={lam}", _powell_fit(train_rows, grad_accum_steps, backward_factor, base_cal, maxiter, lam))
        )
    best_tag, best_cal, best_val = None, None, np.inf
    for tag, cal_dict in candidates:
        if accept is not None and not accept(cal_dict):
            continue  # physically-implausible fit: not eligible
        vm = evaluate(val_rows, grad_accum_steps, backward_factor, cal_dict)
        if np.isfinite(vm) and vm < best_val:
            best_tag, best_cal, best_val = tag, cal_dict, vm
    if best_cal is None:  # no eligible candidate had a finite validation metric -> keep the un-refit prior
        best_tag, best_cal = "none(v2)", copy.deepcopy(base_cal)
    return best_cal, {"choice": best_tag, "val_mdape": float(best_val)}


# ============================== multi-GPU correction fit ==============================
def fit_count(trace: pd.DataFrame, n: int, base_cal: dict) -> list[float]:
    """Return the per-row ratios prediction(mgc[n]=1)/measured for valid, resolvable rows
    at `n` total GPUs. mgc[n] is the median of these ratios. (The trace must carry a ``tot``
    column = number_gpus * number_nodes.)"""
    cal_n = copy.deepcopy(base_cal)
    cal_n["multi_gpu_correction"]["by_num_gpus"][str(n)] = 1.0  # neutralise this count
    rows = trace[trace["tot"] == n]
    ratios: list[float] = []
    with calibration_override(cal_n):
        for r in rows.itertuples():
            if r.model_name not in LLM_SPEC_LIBRARY or str(r.gpu_model) not in GPU_SPEC_LIBRARY:
                continue
            try:
                pred1 = simulate_training_step(
                    model_name=r.model_name,
                    gpu_model=str(r.gpu_model),
                    tokens_per_sample=int(r.tokens_per_sample),
                    batch_size=int(r.batch_size),
                    method=str(r.method),
                    num_gpus=int(n),
                    num_nodes=int(r.number_nodes),
                )["tokens_per_second"]
            except Exception:
                continue
            measured = float(r.dataset_tokens_per_second)
            if measured > 0 and pred1 > 0:
                ratios.append(pred1 / measured)
    return ratios


# ============================ from-scratch table assembly ============================
def _is_physical(cal: dict) -> bool:
    """True iff comm_scale and all mfu_multipliers are inside the physical band -- the guard the
    Tier-1 selector uses to skip a degenerate (non-identifiable) unregularized fit."""
    if not (_SCALE_LO <= cal["comm_scale"] <= _SCALE_HI):
        return False
    return all(_SCALE_LO <= v <= _SCALE_HI for v in cal["mfu_multiplier"].values())


def _neutral_base(reference: dict, models: list[str] | None = None) -> dict:
    """Raw (uncalibrated) kavier expressed as a calibration dict.

    Keeps the reference's STRUCTURE -- which GPUs / methods / models / gpu-counts exist, and the
    schema/version -- but resets every multiplicative correction to a neutral 1.0 and empties
    interaction_scale. The two raw-physics constants (mfu_batch_scale, training_overhead_s) are
    NOT fits, so they are kept as-is. This neutral dict is the prior the from-scratch fit starts from.

    When ``models`` is given, model_scale is restricted to those models (preserving the reference's
    key order), so the fitted table covers exactly the requested set -- e.g. the dense-4 table does
    not carry the two granite-3.1 models. ``models=None`` keeps every model (the all-6 default).
    """
    base = copy.deepcopy(reference)
    base["comm_scale"] = 1.0
    if models is not None:
        keep = set(models)
        base["model_scale"] = {k: v for k, v in base["model_scale"].items() if k in keep}
    for table in ("mfu_multiplier", "method_scale", "model_scale"):
        for key in base[table]:
            base[table][key] = 1.0
    for key in base["multi_gpu_correction"]["by_num_gpus"]:
        base["multi_gpu_correction"]["by_num_gpus"][key] = 1.0
    base["interaction_scale"] = {}
    return base


def _require_columns(trace: pd.DataFrame) -> None:
    """Raise a clear ValueError naming every REQUIRED_COLUMNS entry missing from ``trace``. This is the
    single HARD failure of the data-driven fit: without these columns the mask/target are undefined, so
    we fail loudly and specifically rather than let a downstream KeyError obscure which column is absent."""
    missing = [c for c in REQUIRED_COLUMNS if c not in trace.columns]
    if missing:
        raise ValueError(
            "profiling data is missing required column(s): "
            + ", ".join(missing)
            + f" (required: {', '.join(REQUIRED_COLUMNS)})"
        )


def _filter_valid_rows(
    trace: pd.DataFrame, models: list[str] | None = None, max_total_gpus: int | None = None
) -> pd.DataFrame:
    """The valid, positive-throughput rows the fit covers, filtered to ``models`` when given and to
    ``total_gpus <= max_total_gpus`` when that cap is set (``None`` = keep every GPU count).

    ``regenerate`` passes ``max_total_gpus=8`` (the shipped single-node <=8 fit -- 32/128 come from a
    separate raw-trace mgc step), so the committed table regenerates byte-for-byte. ``calibrate`` passes
    ``None`` so an arbitrary dataset's >8-GPU rows join the main fit directly (see _fit_calibration).

    Shared by _load_profiling (a CSV path) and calibrate (a CSV path or an in-memory DataFrame): both
    apply the *identical* mask so an equivalent input reproduces the same fit. Row ORDER is preserved on
    purpose: the seed-42 split selects rows by position, so re-sorting (or reordering via the filter)
    would change which rows land in train/val/test -- the boolean mask below keeps the original order
    (do not sort). Raises ValueError (via _require_columns) if a required column is absent."""
    _require_columns(trace)
    trace = trace.copy()
    trace["total"] = (pd.to_numeric(trace["number_gpus"]) * pd.to_numeric(trace["number_nodes"])).astype(int)
    mask = (trace["is_valid"] == 1.0) & (trace["dataset_tokens_per_second"] > 0)
    if max_total_gpus is not None:
        mask &= trace["total"] <= max_total_gpus
    if models is not None:
        mask &= trace["model_name"].astype(str).isin(set(models))
    return trace[mask].copy()


def _load_profiling(calib_path: Path, models: list[str] | None = None) -> pd.DataFrame:
    """Read the profiling CSV at ``calib_path`` and return its valid <=8-GPU rows (see
    _filter_valid_rows). When ``models`` is given the rows are filtered to that set (the only
    difference between the 4- and 6-model fits)."""
    trace = pd.read_csv(calib_path, low_memory=False)
    return _filter_valid_rows(trace, models, max_total_gpus=8)


def _row_kw(r: pd.Series) -> dict:
    total = int(r["number_gpus"]) * int(r["number_nodes"])
    return dict(
        model_name=str(r["model_name"]).strip(),
        gpu_model=str(r["gpu_model"]).strip(),
        tokens_per_sample=int(float(r["tokens_per_sample"])),
        batch_size=int(float(r["batch_size"])),
        method=str(r["method"]).strip(),
        num_gpus=total,
        num_nodes=int(r["number_nodes"]),
    )


def _predict(rows: pd.DataFrame, cal_dict: dict) -> np.ndarray:
    # No try/except: a row the engine can't predict should fail loudly, not vanish into a NaN that
    # silently shrinks a median cell (audited: 0 valid profiling rows fail today).
    out = []
    with calibration_override(cal_dict):
        for _, r in rows.iterrows():
            out.append(simulate_training_step(calibrated=True, **_row_kw(r))["tokens_per_second"])
    return np.array(out, dtype=np.float64)


def _ikey(r: pd.Series) -> str:
    k = _row_kw(r)
    return f"{k['model_name']}|{k['method']}|{k['gpu_model']}|{k['num_gpus']}"


def _build_interaction(rows: pd.DataFrame, base_cal: dict) -> dict:
    """Per-config median residual (interaction_scale), rounded to 4dp, for ALL models present in
    ``rows``. ``base_cal`` carries the from-scratch Tier-1 with an EMPTY interaction_scale, so each
    ratio measures exactly what physics + the Tier-1 scales still leave on the table for that cell."""
    pred = _predict(rows, base_cal)
    y = pd.to_numeric(rows["dataset_tokens_per_second"], errors="coerce").to_numpy(np.float64)
    df = pd.DataFrame({"key": [_ikey(r) for _, r in rows.iterrows()], "ratio": y / pred})
    df = df[np.isfinite(df["ratio"]) & (df["ratio"] > 0)]
    return {k: round(float(g["ratio"].median()), 4) for k, g in df.groupby("key", sort=True)}


def _fit_mgc_high(raw_path: Path, frozen_cal: dict, counts: tuple[int, ...] = (32, 128)) -> dict:
    """mgc-only median-ratio fit for the multi-node counts, every other scale frozen at the
    from-scratch <=8 calibration (a joint refit up here is non-identifiable). For each count n:
    mgc[n] = median over resolvable n-GPU rows of prediction(mgc[n]=1) / measured. Returns
    {str(n): value} for every count that had resolvable rows."""
    raw = pd.read_csv(raw_path, low_memory=False)
    raw = raw[(raw["is_valid"] == 1.0) & (raw["dataset_tokens_per_second"] > 0)].copy()
    raw["tot"] = (pd.to_numeric(raw["number_gpus"]) * pd.to_numeric(raw["number_nodes"])).astype(int)
    fitted: dict[str, float] = {}
    for n in counts:
        ratios = fit_count(raw, n, frozen_cal)
        if ratios:
            fitted[str(n)] = round(float(np.median(ratios)), 4)
    return fitted


def _mgc_with_interpolated_gaps(mgc_fitted: dict[str, float], fit_counts: set[int]) -> dict[str, float]:
    """Assemble the multi_gpu_correction table when EVERY GPU count came from the one joint Tier-1
    fit (the calibrate() >8-in-main-fit path). ``mgc_fitted`` is tier1's by_num_gpus (the template's
    key set; counts the Powell layout varied hold fitted values, the rest are still neutral 1.0).
    ``fit_counts`` are exactly the counts Powell varied. For each template count: use its fitted value
    if it was fit; otherwise log2-geometric interpolation of the nearest fitted neighbours (this is the
    same interpolation the <=8 path uses for 16/64, generalized), clamping to the single neighbour when
    the gap has no bracketing pair, or neutral 1.0 if nothing was fit at all."""
    keys = sorted(int(k) for k in mgc_fitted)
    fit = sorted(fit_counts)
    out: dict[str, float] = {}
    for k in keys:
        if k in fit_counts:
            out[str(k)] = mgc_fitted[str(k)]  # raw fitted value (unrounded, as the <=8 path uses 2/4/8)
            continue
        lower = [c for c in fit if c < k]
        upper = [c for c in fit if c > k]
        if lower and upper:
            lo, hi = lower[-1], upper[0]
            w = (np.log2(k) - np.log2(lo)) / (np.log2(hi) - np.log2(lo))
            val = float(np.exp((1 - w) * np.log(mgc_fitted[str(lo)]) + w * np.log(mgc_fitted[str(hi)])))
            out[str(k)] = round(val, 2)
        elif lower:
            out[str(k)] = round(float(mgc_fitted[str(lower[-1])]), 2)  # above the top fitted count: hold flat
        elif upper:
            out[str(k)] = round(float(mgc_fitted[str(upper[0])]), 2)  # below the bottom fitted count: hold flat
        else:
            out[str(k)] = 1.0  # no GPU count was ever fit -> neutral
    return out


# Rebuild the WHOLE calibration table from scratch from the measured runs (6 steps below).
def _fit_calibration(
    reference: dict,
    rows: pd.DataFrame,
    raw_path: Path | None,
    models: list[str] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[dict, dict]:
    """The from-scratch two-tier fit given ALREADY-loaded valid ``rows`` (steps 1-6 of the recipe).
    Shared verbatim by regenerate() -- which loads the fixed internal profiling CSV capped to <=8 GPU --
    and calibrate() -- which takes an arbitrary CSV/DataFrame with NO GPU-count cap -- so an equivalent
    input reproduces the same numbers.

    The multi-GPU correction has TWO mutually-exclusive sources, chosen by whether ``rows`` already
    carries >8-GPU data (see Step 5). If it does (calibrate's uncapped rows), mgc for every count comes
    from the single joint Tier-1 fit and ``raw_path`` is NOT consulted -- consulting it too would
    double-count those >8 rows. If it does not (regenerate's <=8 rows, or any <=8-only dataset),
    ``raw_path`` (if it exists) supplies the >8-GPU multi-node rows for the separate mgc 32/128 fit;
    None/absent leaves mgc >8 neutral. ``log`` receives the progress lines. Returns
    (calibration_dict, metrics) -- metrics is a small fit report (row counts, held-out MdAPE, ...)."""
    # Step 1: raw kavier = the reference's structure with every correction neutralised to 1.0
    #         (model_scale restricted to the selected set).
    neutral = _neutral_base(reference, models)

    # Step 2: split the valid rows (selected models) into train / val / test (seed 42).
    train, val, test = train_val_test_split(rows)
    n_models = rows["model_name"].astype(str).nunique()
    max_total = int(rows["total"].max()) if len(rows) else 0
    log(
        f"  rows: train={len(train)} val={len(val)} test={len(test)} "
        f"(valid, tput>0, <={max_total} GPU, {n_models} models)"
    )

    # Step 3: Tier-1 -- regularized Powell joint fit of the global scales (comm_scale, per-GPU MFU,
    #         per-method, per-model, multi-GPU 2/4/8) from the neutral prior; lambda picked on val,
    #         skipping any physically-degenerate (non-identifiable comm_scale) candidate (see _is_physical).
    tier1, info = select_calibration(
        train, val, grad_accum_steps=1, backward_factor=2.0, base_cal=neutral, accept=_is_physical
    )
    log(
        f"  Tier-1 fit: regularisation choice={info['choice']!r} val_mdape={info['val_mdape']:.2f}% "
        f"(comm_scale={tier1['comm_scale']:.3f})"
    )

    # Step 4: Tier-2 -- the leftover per-cell residual (interaction_scale): median(measured / pred)
    #         on train+val, with the Tier-1 scales applied and interaction itself empty.
    base = copy.deepcopy(tier1)
    base["interaction_scale"] = {}
    train_val = pd.concat([train, val], ignore_index=True)
    interaction = _build_interaction(train_val, base)

    # Step 5: the multi-GPU correction. Its source depends on whether >8-GPU rows are already in the
    #         main fit -- and the two sources are MUTUALLY EXCLUSIVE, so a >8 row is never counted twice.
    mgc_lo = tier1["multi_gpu_correction"]["by_num_gpus"]
    high_totals = sorted({int(t) for t in rows["total"].unique() if t > 8})
    if high_totals:
        # calibrate()'s uncapped path: the >8-GPU rows were part of the joint Tier-1 Powell fit, which
        # therefore already calibrated mgc for every GPU count PRESENT in the train split (2/4/8 AND the
        # >8 counts) in a single pass. We take mgc straight from that joint fit. We DO NOT also run the
        # separate raw-trace _fit_mgc_high step here: that step exists only to REACH >8 counts the main
        # fit never saw; running it now would (a) double-count the >8 rows -- once in the joint fit, once
        # in the frozen median-ratio refit -- and (b) be internally inconsistent, since that refit divides
        # comm_scale back out of a comm_scale the very same rows just helped fit. Template counts absent
        # from the data (e.g. 16/64 when the trace has 32/128 but not 16/64) are filled by the same
        # log2-geometric interpolation the <=8 path uses for 16/64.
        fitted_counts = {int(key) for (kind, key) in _vary_layout(train, neutral) if kind == "mgc"}
        mgc = _mgc_with_interpolated_gaps(mgc_lo, fitted_counts)
        mgc_note = (
            "Every GPU count present in the data was calibrated jointly by the Tier-1 Powell fit: the "
            ">8-GPU rows are part of the main fit, NOT a separate raw-trace step, so no >8 row is "
            f"double-counted. Counts fit directly: {sorted(fitted_counts)}. Template counts absent from "
            "the data are log2-geometric interpolations of the fitted neighbours (as the <=8 path does "
            "for 16/64). interaction_scale now covers whatever >8 cells the data provides."
        )
        log(f"  mgc: joint Tier-1 fit on GPU counts {sorted(fitted_counts)} (>8 in the main fit); gaps interpolated")
    elif raw_path is not None and raw_path.exists():
        frozen = copy.deepcopy(tier1)
        frozen["interaction_scale"] = interaction  # no effect >8 GPU (no keys there); kept for consistency
        hi = _fit_mgc_high(raw_path, frozen, counts=(32, 128))
        if "32" not in hi or "128" not in hi:
            raise SystemExit(f"raw multi-node trace {raw_path} has no resolvable 32/128-GPU rows")
        mgc = {
            "2": mgc_lo["2"],
            "4": mgc_lo["4"],
            "8": mgc_lo["8"],
            "16": round((mgc_lo["8"] * hi["32"]) ** 0.5, 2),
            "32": hi["32"],
            "64": round((hi["32"] * hi["128"]) ** 0.5, 2),
            "128": hi["128"],
        }
        mgc_note = (
            "2/4/8: joint Powell fit on the <=8-GPU profiling rows (all models, single-node). "
            "32/128: mgc-only median-ratio fit on the raw multi-node trace, all other scales frozen "
            "at the from-scratch <=8 calibration (a joint refit up here is non-identifiable). "
            "16/64: log2-geometric interpolation of neighbours. >8 is a scalar extrapolation: no "
            "interaction_scale coverage above 8 GPU; the recommender restricts to <=8."
        )
    else:
        mgc = {"2": mgc_lo["2"], "4": mgc_lo["4"], "8": mgc_lo["8"], "16": 1.0, "32": 1.0, "64": 1.0, "128": 1.0}
        mgc_note = (
            "2/4/8: joint Powell fit on the <=8-GPU profiling rows. 16/32/64/128 left NEUTRAL (1.0): "
            "the raw multi-node trace was unavailable at regeneration time, so >8 GPU is uncalibrated; "
            "the recommender restricts to <=8."
        )
        log(f"  WARNING: raw multi-node trace not found at {raw_path}; mgc >8 left neutral (1.0)")

    # Step 6: assemble in the shipped key order + format. Physics constants + schema/version are kept
    #         from the reference template; every scale below was fit from the data in steps 3-5.
    out = {
        "schema_version": reference["schema_version"],
        "version": reference["version"],
        "comm_scale": tier1["comm_scale"],
        "training_overhead_s": reference["training_overhead_s"],
        "mfu_batch_scale": reference["mfu_batch_scale"],
        "mfu_multiplier": tier1["mfu_multiplier"],
        "multi_gpu_correction": {"by_num_gpus": mgc, "_note": mgc_note},
        "method_scale": tier1["method_scale"],
        "model_scale": tier1["model_scale"],
        "interaction_scale": interaction,
        "_note": (
            "Regenerated from scratch by kavier.sdk.training.calibration.engine: raw (uncalibrated) "
            "kavier + the profiling trace, seed-42 70/15/15 split. Tier-1 (comm_scale / per-GPU "
            "MFU / per-method / per-model / multi-GPU 2-4-8) = regularized Powell joint fit on train, "
            "lambda chosen on val; Tier-2 interaction_scale = per-cell median residual on train+val. "
            "Nothing carried from any prior calibration; the held-out 15% test split reports the accuracy."
        ),
    }

    # Report the held-out accuracy of the assembled table (the test split never touched the fit).
    held_out = evaluate(test, 1, 2.0, out)
    raw_md = evaluate(test, 1, 2.0, neutral)
    log(f"  held-out test MdAPE: from-scratch={held_out:.2f}%  (raw/uncalibrated={raw_md:.2f}%)")
    metrics = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_models": int(n_models),
        "tier1_choice": info["choice"],
        "val_mdape": float(info["val_mdape"]),
        "held_out_mdape": float(held_out),
        "raw_mdape": float(raw_md),
    }
    return out, metrics


def regenerate(
    reference: dict, profiling_dir: Path, raw_trace: Path | None = None, models: list[str] | None = None
) -> dict:
    """Rebuild the whole table from scratch. ``models`` selects the model-set the Tier-1/Tier-2 fit
    covers (the profiling trace is filtered to it); ``None`` keeps every model (the all-6 default that
    reproduces calibration.json). The trace is capped at <=8 GPU (via ``_load_profiling``), so the
    >8-GPU mgc always comes from the separate raw multi-node trace -- the multi-GPU correction is a
    global communication-scaling scalar, not a per-model one, and the dense-4 models have no multi-node
    rows of their own. This <=8 cap is what keeps regeneration byte-identical; the data-driven
    ``calibrate`` uses NO cap (see its docstring)."""
    calib = profiling_dir / PROFILING_CSV
    if not calib.exists():
        raise SystemExit(f"profiling trace not found: {calib}")
    rows = _load_profiling(calib, models)
    if rows.empty:
        raise SystemExit(f"no valid <=8-GPU rows for models={models} in {calib}")
    raw_path = raw_trace or (profiling_dir / RAW_MULTINODE_CSV)
    out, _metrics = _fit_calibration(reference, rows, raw_path, models)
    return out


# ============================ parameterized (data-driven) calibrate ============================
# Minimum valid rows a model needs to be AUTO-selected when calibrate(models=None). Every shipped
# calibrated model clears this comfortably (the smallest, granite-3-8b, has 46). An explicit models=
# list bypasses this floor (the caller has asked for exactly those, however thin).
_MIN_ROWS_TO_AUTOSELECT = 8

# Minimum valid rows a (requested) model needs to be FIT at all: the seed-42 3-way split needs a
# non-empty train/val/test, which requires >=3 rows. A requested model below this is skipped (with a
# warning), never fit on an empty split. This is separate from the suitability floor MIN_ROWS_PER_MODEL
# (=30), which only WARNS about a thin-but-fittable model; a model can be fittable yet unsuitable.
_MIN_ROWS_TO_FIT = 3


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _select_models(valid_rows: pd.DataFrame, min_rows: int = _MIN_ROWS_TO_AUTOSELECT) -> list[str]:
    """Model names present in ``valid_rows`` with at least ``min_rows`` rows, sorted -- the auto-set
    calibrate fits when no explicit models list is given."""
    counts = valid_rows["model_name"].astype(str).value_counts()
    return sorted(str(m) for m, c in counts.items() if int(c) >= min_rows)


def _suitability_report(valid: pd.DataFrame, models: list[str]) -> str | None:
    """Check the post-filter data for the fitted ``models`` against the 'suitable calibration dataset'
    properties and return ONE multi-line warning naming only the checks that FAIL (with the offending
    model / GPU-cell lists), or None if the data looks suitable. Advisory ONLY -- it never changes the
    fit, so an unsuitable dataset is still fit, just flagged (the headline warning of `kavier calibrate`)."""
    df = valid[valid["model_name"].astype(str).isin(set(models))].copy()
    df["model_name"] = df["model_name"].astype(str)
    df["gpu_model"] = df["gpu_model"].astype(str)
    problems: list[str] = []

    # (a) >= MIN_ROWS_PER_MODEL valid rows per fitted model.
    counts = df["model_name"].value_counts()
    thin = sorted(m for m in models if int(counts.get(m, 0)) < MIN_ROWS_PER_MODEL)
    if thin:
        problems.append(
            f"    - fewer than {MIN_ROWS_PER_MODEL} valid rows for model(s): "
            + ", ".join(f"{m} (n={int(counts.get(m, 0))})" for m in thin)
        )

    # (b) a spread of batch sizes per (model, GPU) cell.
    per_cell = df.groupby(["model_name", "gpu_model"])["batch_size"].nunique()
    flat = sorted(
        f"{m}/{g} ({int(n)} batch size)" for (m, g), n in per_cell.items() if int(n) < MIN_DISTINCT_BATCH_SIZES
    )
    if flat:
        problems.append(
            f"    - fewer than {MIN_DISTINCT_BATCH_SIZES} distinct batch sizes in (model, GPU) cell(s): "
            + ", ".join(flat)
        )

    # (c) a spread of total-GPU counts (needed to identify the multi-GPU correction).
    n_counts = int(df["total"].nunique())
    if n_counts < MIN_DISTINCT_GPU_COUNTS:
        present = sorted(int(t) for t in df["total"].unique())
        problems.append(
            f"    - only {n_counts} distinct total-GPU count(s) present ({present}); "
            f">= {MIN_DISTINCT_GPU_COUNTS} needed to fit multi-GPU corrections"
        )

    if not problems:
        return None
    return (
        "Tuning may have produced poor results. A suitable calibration dataset should have:\n"
        f"  - the required columns; >= {MIN_ROWS_PER_MODEL} valid rows per model; a spread of batch sizes "
        "per (model, GPU);\n"
        "  - a spread of GPU counts if you want multi-GPU corrections; coverage of the models/GPUs/methods.\n"
        "This dataset falls short on:\n" + "\n".join(problems)
    )


def calibrate(source: str | Path | pd.DataFrame, models: list[str] | None = None) -> dict:
    """Fit a calibration table FROM SCRATCH on ``source`` -- a profiling CSV path or an in-memory
    DataFrame carrying the profiling columns -- using the SAME two-tier (regularized Powell + per-cell
    interaction_scale) recipe as regenerate(); this is that recipe parameterized on the given data
    instead of the fixed internal trace.

    The fit keeps only ``is_valid == 1`` and ``dataset_tokens_per_second > 0`` rows (targeting
    ``dataset_tokens_per_second``) at ANY GPU count -- unlike regenerate(), calibrate applies NO
    total-GPU cap, so a dataset's >8-GPU rows join the main joint fit directly (the multi_gpu_correction
    for every count present then comes from that one fit; see _fit_calibration Step 5). Because it no
    longer runs the separate <=8 + raw-trace two-step, ``calibrate(profiling_trace.csv)`` does NOT
    reproduce the shipped calibration.json byte-for-byte (regenerate() does).

    ``models`` restricts the fit to those model names; None (default) auto-selects every model with at
    least ``_MIN_ROWS_TO_AUTOSELECT`` valid rows. Robustness: a missing REQUIRED_COLUMNS column is the
    only hard failure (a clear ValueError naming it); a requested model too thin to fit
    (< ``_MIN_ROWS_TO_FIT`` rows) is SKIPPED with a warning, not an error; and if the filtered data
    violates any suitability property (see _suitability_report) a single headline warning is emitted --
    the fit still runs on whatever the data supports. When ``source`` is a path a sibling
    ``raw_trace.csv`` is consulted for the >8-GPU mgc ONLY when the data itself has no >8 rows.

    The shipped calibration.json is the STRUCTURAL template only (which GPUs/methods exist, the
    raw-physics constants mfu_batch_scale/training_overhead_s, and schema/version); none of its fitted
    scales are carried -- the fit starts from a neutral 1.0 prior, and a neutral model_scale prior is
    seeded for any requested model the template does not already cover. Returns the calibration dict
    (same schema as calibration.json); serialize it with ``_dumps`` to match the shipped file format."""
    reference = json.loads(CAL_PATH.read_text(encoding="utf-8"))

    if isinstance(source, pd.DataFrame):
        trace = source
        raw_path: Path | None = None
    else:
        src = Path(source)
        trace = pd.read_csv(src, low_memory=False)
        sibling = src.parent / RAW_MULTINODE_CSV
        raw_path = sibling if sibling.exists() else None

    # Uncapped: keep every valid row at ANY GPU count (raises ValueError if a required column is absent).
    valid = _filter_valid_rows(trace, None, max_total_gpus=None)
    if valid.empty:
        raise SystemExit("no valid rows (is_valid==1, dataset_tokens_per_second>0) in the input")

    models_final = list(models) if models is not None else _select_models(valid)
    if not models_final:
        raise SystemExit(
            f"no model has >= {_MIN_ROWS_TO_AUTOSELECT} valid rows to auto-fit; pass models= to fit a specific set"
        )

    # A requested model too thin to fit (below the 3-way split floor) is skipped with a warning, not an
    # error -- the remaining models still fit. (Auto-selected models already clear _MIN_ROWS_TO_AUTOSELECT.)
    counts_all = valid["model_name"].astype(str).value_counts()
    fittable = [m for m in models_final if int(counts_all.get(m, 0)) >= _MIN_ROWS_TO_FIT]
    skipped = [m for m in models_final if m not in fittable]
    if skipped:
        warnings.warn(
            f"skipping model(s) with fewer than {_MIN_ROWS_TO_FIT} valid rows (too few to fit): "
            + ", ".join(f"{m} (n={int(counts_all.get(m, 0))})" for m in skipped),
            UserWarning,
            stacklevel=2,
        )
    models_final = fittable
    if not models_final:
        raise SystemExit(f"no requested model has >= {_MIN_ROWS_TO_FIT} valid rows to fit (all skipped as too thin)")

    # Headline suitability warning: emitted once, naming only the checks that fail; silent (one info
    # line) when the data looks fine. Advisory -- it never changes what is fit.
    report = _suitability_report(valid, models_final)
    if report is not None:
        warnings.warn(report, UserWarning, stacklevel=2)
    else:
        _eprint("dataset looks suitable")

    rows = _filter_valid_rows(trace, models_final, max_total_gpus=None)
    if rows.empty:
        raise SystemExit(f"no valid rows for models={models_final} in the input")

    # Seed a neutral (1.0) prior for any model / GPU / method PRESENT in the data but ABSENT from the
    # template, so calibrate() fits an ARBITRARY dataset: a novel GPU or training method is then
    # calibrated from its own rows instead of KeyError-crashing in the Powell layout lookup (_get). The
    # shipped regenerate() path never runs this -- its fixed trace uses only template keys -- so the
    # byte-identity of the shipped tables is unaffected. (For the shipped 6-model trace nothing is
    # missing, so the template is used unchanged.)
    reference = copy.deepcopy(reference)
    for m in sorted(set(rows["model_name"].astype(str)) - set(reference["model_scale"])):
        reference["model_scale"][m] = 1.0
    for g in sorted(set(rows["gpu_model"].astype(str)) - set(reference["mfu_multiplier"])):
        reference["mfu_multiplier"][g] = 1.0
    for meth in sorted(set(rows["method"].astype(str)) - set(reference["method_scale"])):
        reference["method_scale"][meth] = 1.0

    out, _metrics = _fit_calibration(reference, rows, raw_path, models_final, log=_eprint)
    return out


def _dumps(cal_dict: dict) -> str:
    """Serialize exactly as the shipped file is written (indent=2 + trailing newline)."""
    return json.dumps(cal_dict, indent=2) + "\n"


# ======================================== CLI ========================================
def _resolve_models(model_set: str, models_csv: str | None) -> list[str]:
    """The model list to fit: an explicit --models CSV wins, else the named --model-set."""
    if models_csv:
        return [m.strip() for m in models_csv.split(",") if m.strip()]
    return MODEL_SETS[model_set]


def _print_diff(expected: str, actual: str, *, fromfile: str, tofile: str) -> None:
    sys.stdout.writelines(
        difflib.unified_diff(
            expected.splitlines(keepends=True), actual.splitlines(keepends=True), fromfile=fromfile, tofile=tofile
        )
    )


def _write_set(model_set: str, text: str) -> None:
    """Write the selected set's versions/ file; the 6-model set also mirrors the root calibration.json
    (the shipped default the accessor/coastline/runtime load)."""
    VERSIONS_DIR.mkdir(exist_ok=True)
    target = VERSION_FILES[model_set]
    target.write_text(text)
    print(f"  wrote {target.relative_to(REPO_ROOT)}")
    if model_set == "6":
        CAL_PATH.write_text(text)
        print(f"  wrote {CAL_PATH.relative_to(REPO_ROOT)} (the default = 6-model)")


def _check_both(reference: dict, args: argparse.Namespace) -> None:
    """Rebuild BOTH model-sets and assert each reproduces its versions/ file byte-for-byte, and that
    calibration.json equals the 6-model file. The determinism / self-consistency guard for v0.4."""
    print(f"--check: rebuilding BOTH model-sets from {args.profiling_data} (template: {args.reference})")
    ok = True
    for model_set in ("6", "4"):
        models = MODEL_SETS[model_set]
        print(f"\n[{model_set}-model] models={models}")
        new_text = _dumps(regenerate(reference, args.profiling_data, raw_trace=args.raw_trace, models=models))
        target = VERSION_FILES[model_set]
        if not target.exists():
            print(f"  MISSING: {target} (run: --model-set {model_set} --write)")
            ok = False
            continue
        target_text = target.read_text(encoding="utf-8")
        identical = new_text == target_text
        print(f"  reproduces {target.name} byte-for-byte: {identical}")
        if not identical:
            ok = False
            _print_diff(target_text, new_text, fromfile=target.name, tofile="regenerated")

    root_eq_6 = CAL_PATH.read_text(encoding="utf-8") == VERSION_FILES["6"].read_text(encoding="utf-8")
    print(f"\ncalibration.json == versions/{VERSION_FILES['6'].name}: {root_eq_6}")
    ok = ok and root_eq_6

    if not ok:
        raise SystemExit(
            "CHECK FAILED: a model-set did not reproduce its versions/ file byte-for-byte "
            "(re-run --write if the fit legitimately changed, else investigate non-determinism)"
        )
    print("\nCHECK PASSED: both model-sets reproduce their versions/ files byte-for-byte")


def _cmd_regen(reference: dict, args: argparse.Namespace) -> None:
    """The from-scratch (re)generation flow for the selected --model-set (--write/--out/--snapshot)."""
    models = _resolve_models(args.model_set, args.regen_models)
    print(
        f"regenerating from scratch from {args.profiling_data} (structural template: {args.reference}; models={models})"
    )
    regenerated = regenerate(reference, args.profiling_data, raw_trace=args.raw_trace, models=models)
    new_text = _dumps(regenerated)

    print(
        f"  fit from data: comm_scale, mfu_multiplier (per GPU), method_scale, model_scale (all "
        f"{len(regenerated['model_scale'])} models), mgc 2/4/8 (Powell) + 32/128 (raw trace); "
        f"interaction_scale ({len(regenerated['interaction_scale'])} cells); mgc 16/64 by formula"
    )
    print("  kept from template (raw physics, never fitted): mfu_batch_scale, training_overhead_s, schema/version")

    # If the selected set has a versions/ target, report (and diff) byte-identity against it.
    target = None if args.regen_models else VERSION_FILES.get(args.model_set)
    if target is not None and target.exists():
        target_text = target.read_text(encoding="utf-8")
        identical = new_text == target_text
        print(f"  reproduces {target.name} byte-for-byte: {identical}")
        if not identical:
            _print_diff(target_text, new_text, fromfile=target.name, tofile="regenerated")

    if args.out:
        args.out.write_text(new_text)
        print(f"  wrote {args.out}")
    if args.snapshot:
        snap_dir = CAL_PATH.parent / "snapshots"
        snap_dir.mkdir(exist_ok=True)
        snap = snap_dir / f"calibration-{datetime.now():%Y%m%d-%H%M%S}.json"
        snap.write_text(new_text)
        print(f"  wrote snapshot {snap.relative_to(REPO_ROOT)}")
        print(f"  compare with:  diff {CAL_PATH} {snap}")
    if args.write:
        _write_set(args.model_set, new_text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument(
        "--profiling-data",
        type=Path,
        default=TRACE_ARCHIVE,
        help="directory holding the profiling-dataset CSVs (default: the in-repo trace-archive)",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        default=CAL_PATH,
        help="calibration.json used ONLY as the structural template (which GPUs/methods/models/"
        "gpu-counts exist) + the raw-physics constants (mfu_batch_scale, training_overhead_s) + "
        "schema/version; NONE of its fitted scales are carried (default: the committed file)",
    )
    ap.add_argument(
        "--raw-trace",
        type=Path,
        default=None,
        help="raw multi-node trace for the >8-GPU mgc fit (default: <profiling-data>/raw_trace.csv)",
    )
    ap.add_argument(
        "--model-set",
        choices=sorted(MODEL_SETS),
        default="6",
        help="which model-set to (re)fit for --write/--out/--snapshot/the default report "
        "(default: 6 = the dense-4 + granite-3.1 set that is the shipped calibration.json); "
        "--check always rebuilds both",
    )
    ap.add_argument(
        "--models", dest="regen_models", help="comma-separated model list overriding --model-set (advanced)"
    )
    ap.add_argument("--out", type=Path, help="write the regenerated calibration to this path")
    ap.add_argument(
        "--snapshot",
        action="store_true",
        help="also save the rebuilt table to a timestamped calibration-YYYYMMDD-HHMMSS.json under the "
        "package's snapshots/ folder, so you can diff it against the committed one; never overwrites it",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="rebuild the selected --model-set and write versions/calibration_<n>model.json "
        "(the 6-model set also overwrites the root calibration.json default)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="rebuild BOTH model-sets and assert each reproduces its versions/ file byte-for-byte "
        "(and that calibration.json == the 6-model file); a determinism / self-consistency guard; exit 1 if not",
    )

    args = ap.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if args.check:
        _check_both(reference, args)
        return
    _cmd_regen(reference, args)


if __name__ == "__main__":
    main()
