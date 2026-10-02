#!/usr/bin/env python
"""Fit the calibration tables (calibration.json and versions/) from measured runs. Dev-only.

Kavier predicts training speed from physics. Calibration is a small set of correction factors,
learned from measured runs, that bring those predictions closer to measurements (physics-only ~16%
error, calibrated ~10%). This module rebuilds them from the data and carries nothing over from a
previous calibration. The runtime accessor is in __init__.py.

The fit has two inputs:
  1. raw (uncalibrated) kavier: the physics, with every correction set to a neutral 1.0
  2. the measured profiling runs: trace-archive/profiling-dataset/profiling_trace.csv

Two tiers (see ``regenerate``):
  - Tier 1 (global scales): Powell minimization in log-space (derivative-free) of median APE plus an
    L2 pull toward neutral, with lambda chosen on the validation split.
  - Tier 2 (per-config): median of measured/predicted ratios.
  The >8-GPU mgc at 32/128 is the same median ratio on the multi-node trace; 16/64 are log2 geometric
  means. The held-out 15% test split is never fit and only reports accuracy.

Two model sets use the same recipe, with the profiling trace filtered to the set:
  - 4-model (dense-4: mistral-7b-v0.1, granite-3.3-8b, granite-3-8b, llama3.2-3b) -> versions/calibration_4model.json
  - 6-model (dense-4 + granite-3.1-2b + granite-3.1-8b-instruct)                  -> versions/calibration_6model.json
The 6-model fit is the shipped default; calibration.json and calibration_6model.json are byte-identical.

Usage (ENG = kavier.sdk.training.calibration.engine):
  python -m ENG --check                # rebuild both sets; confirm each matches its versions/ file
  python -m ENG --write                # rebuild --model-set (default 6) + overwrite its file(s)
  python -m ENG --model-set 4 --write  # rebuild + write only the 4-model file
  python -m ENG --snapshot             # rebuild + save a timestamped copy to diff

The measured-run files are internal and not shipped; pass them with --profiling-data / --raw-trace.
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
# parents[4] = src, [5] = repo root, [6] = workspace holding trace-archive/.
_HERE = Path(__file__).resolve()
SRC = _HERE.parents[4]
REPO_ROOT = _HERE.parents[5]
WORKSPACE = _HERE.parents[6]
TRACE_ARCHIVE = WORKSPACE / "trace-archive" / "profiling-dataset"
CAL_PATH = _HERE.parent / "calibration.json"

# Merged profiling export: curated dense-4 rows across GPU types plus the controlled-benchmark granite rows.
# The only input of the <=8-GPU Tier 1 and Tier 2 fit.
PROFILING_CSV = "profiling_trace.csv"
# Raw multi-node trace, read only to fit the >8-GPU multi-GPU correction (32/128 GPUs).
RAW_MULTINODE_CSV = "raw_trace.csv"

# Plausibility band for Tier 1 candidates. At <=8 GPUs on one node comm is small next to compute, so the
# data barely constrain comm_scale or, under weak regularization, mfu_multiplier. With lambda=0 comm_scale
# fits noise at ~20x: similar test error, but not physical, and it corrupts the >8-GPU mgc fit, which
# divides comm back out. Held-out test MdAPE with the band is ~10% (comm_scale ~1.23).
_SCALE_LO, _SCALE_HI = 0.5, 2.0

# Model sets differ only in the profiling rows they fit. DENSE_4 is the Exp1 head-to-head set; ALL_6 adds
# the two controlled-benchmark granite-3.1 models (Exp4 in-vitro) and reproduces calibration.json.
DENSE_4 = ["mistral-7b-v0.1", "granite-3.3-8b", "granite-3-8b", "llama3.2-3b"]
ALL_6 = [*DENSE_4, "granite-3.1-2b", "granite-3.1-8b-instruct"]
MODEL_SETS: dict[str, list[str]] = {"4": DENSE_4, "6": ALL_6}

# calibration.json (root) == versions/calibration_6model.json.
VERSIONS_DIR = CAL_PATH.parent / "versions"
VERSION_FILES: dict[str, Path] = {
    "4": VERSIONS_DIR / "calibration_4model.json",
    "6": VERSIONS_DIR / "calibration_6model.json",
}

# Seed-42 70/15/15 split. train_test_split permutes by row count and seed but selects by position, so
# byte-identical regeneration needs the canonical curated CSV in its original row order. Do not sort it.
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.176  # 0.176 of the remaining 85% ~= 15% of total

# Columns every profiling trace needs. A missing one is the only hard failure of calibrate() (ValueError
# naming the columns); thin or narrow data only warns. Callers can use this to pre-check a trace.
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

# Thresholds for the suitability warning in calibrate(); they do not change the fit (see _suitability_report).
# MIN_ROWS_PER_MODEL: below this a model's scales are poorly pinned.
# MIN_DISTINCT_BATCH_SIZES: one batch size per (model, GPU) leaves the batch/MFU curve unconstrained.
# MIN_DISTINCT_GPU_COUNTS: with one total-GPU count the multi-GPU correction is unidentifiable.
MIN_ROWS_PER_MODEL = 30
MIN_DISTINCT_BATCH_SIZES = 2
MIN_DISTINCT_GPU_COUNTS = 2


@contextmanager
def calibration_override(cal_dict: dict[str, Any]) -> Iterator[None]:
    """Install ``cal_dict`` as the live calibration for the with-block, then restore the previous one.

    Swaps the module global kavier.sdk.training.calibration._CAL, the same swap Coastline uses."""
    import kavier.sdk.training.calibration as cal

    saved = cal._CAL
    cal._CAL = cal_dict
    try:
        yield
    finally:
        cal._CAL = saved


# ==================================== metrics =====================================
def mdape(measured: np.ndarray, pred: np.ndarray) -> float:
    """Return the median absolute percentage error [%] over rows with ``measured > 0``."""
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
    """Split ``df`` into (train, val, test), seed-42 70/15/15 (see the comment at SEED)."""
    temp, test = train_test_split(df, test_size=test_size, random_state=seed)
    train, val = train_test_split(temp, test_size=val_size, random_state=seed)
    return train, val, test


# ================================ Tier-1 Powell fit ================================
# Regularized Powell refit of the global scales (comm_scale, mfu_multiplier, existing multi_gpu_correction
# keys, method_scale, model_scale) in log-space: train MdAPE + an L2 pull toward the prior.
# select_calibration picks lambda on validation; regenerate() starts from neutral raw kavier.
_SHIPPED = json.loads(CAL_PATH.read_text())


def _powell_row_args(rows: pd.DataFrame) -> list[dict]:
    """Return per-row engine kwargs, extracted once for repeated evaluation."""
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
    """Return the (kind, key) scale entries the train rows exercise; the rest stay at base_cal.

    Only mgc keys already in the table vary; other GPU totals keep the engine fallback."""
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
    """Return the MdAPE [%] of ``cal_dict`` on ``rows``."""
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
    """Fit each lambda on train, add base_cal without a refit, and keep the best validation MdAPE.

    With ``accept``, only candidates where ``accept(cal)`` is True are eligible; this rejects degenerate
    fits such as a runaway comm_scale under weak regularization. Returns (calibration, {choice, val_mdape})."""
    candidates = [("none(v2)", copy.deepcopy(base_cal))]
    for lam in lambdas:
        candidates.append(
            (f"lam={lam}", _powell_fit(train_rows, grad_accum_steps, backward_factor, base_cal, maxiter, lam))
        )
    best_tag, best_cal, best_val = None, None, np.inf
    for tag, cal_dict in candidates:
        if accept is not None and not accept(cal_dict):
            continue
        vm = evaluate(val_rows, grad_accum_steps, backward_factor, cal_dict)
        if np.isfinite(vm) and vm < best_val:
            best_tag, best_cal, best_val = tag, cal_dict, vm
    if best_cal is None:  # no eligible candidate with a finite metric: keep the prior
        best_tag, best_cal = "none(v2)", copy.deepcopy(base_cal)
    return best_cal, {"choice": best_tag, "val_mdape": float(best_val)}


# ============================== multi-GPU correction fit ==============================
def fit_count(trace: pd.DataFrame, n: int, base_cal: dict) -> list[float]:
    """Return prediction(mgc[n]=1) / measured for valid, resolvable rows at ``n`` total GPUs.

    mgc[n] is the median of these ratios. ``trace`` needs a ``tot`` column = number_gpus * number_nodes."""
    cal_n = copy.deepcopy(base_cal)
    cal_n["multi_gpu_correction"]["by_num_gpus"][str(n)] = 1.0
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
    """Return True if comm_scale and every mfu_multiplier lie in [_SCALE_LO, _SCALE_HI]."""
    if not (_SCALE_LO <= cal["comm_scale"] <= _SCALE_HI):
        return False
    return all(_SCALE_LO <= v <= _SCALE_HI for v in cal["mfu_multiplier"].values())


def _neutral_base(reference: dict, models: list[str] | None = None) -> dict:
    """Return raw (uncalibrated) kavier as a calibration dict, the prior of the from-scratch fit.

    Keeps the reference's structure (GPUs, methods, models, GPU counts, schema/version), resets every
    multiplicative correction to 1.0 and empties interaction_scale. mfu_batch_scale and
    training_overhead_s are raw-physics constants and are kept. ``models`` restricts model_scale to that
    set in the original key order; None keeps every model.
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
    """Raise ValueError naming every REQUIRED_COLUMNS entry missing from ``trace``.

    The one hard failure of the data-driven fit; a later KeyError would not say which column is missing."""
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
    """Return the valid, positive-throughput rows, filtered to ``models`` and ``total <= max_total_gpus``.

    None disables either filter. regenerate() passes ``max_total_gpus=8`` (32/128 come from the raw
    trace), so the committed table regenerates byte-for-byte; calibrate() passes None, so >8-GPU rows
    join the main fit. Row order is kept because the seed-42 split selects by position. Raises
    ValueError via _require_columns if a required column is missing."""
    _require_columns(trace)
    # Invalid rows go first: their GPU counts can be NaN, which the int cast rejects.
    trace = trace[(trace["is_valid"] == 1.0) & (trace["dataset_tokens_per_second"] > 0)].copy()
    trace["total"] = (pd.to_numeric(trace["number_gpus"]) * pd.to_numeric(trace["number_nodes"])).astype(int)
    if max_total_gpus is not None:
        trace = trace[trace["total"] <= max_total_gpus]
    if models is not None:
        trace = trace[trace["model_name"].astype(str).isin(set(models))]
    return trace.copy()


def _load_profiling(calib_path: Path, models: list[str] | None = None) -> pd.DataFrame:
    """Read the profiling CSV and return its valid <=8-GPU rows, filtered to ``models`` if given."""
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
    # No try/except, so an unpredictable row raises; a NaN would silently shrink a median cell.
    # Audited: no valid profiling row fails.
    out = []
    with calibration_override(cal_dict):
        for _, r in rows.iterrows():
            out.append(simulate_training_step(calibrated=True, **_row_kw(r))["tokens_per_second"])
    return np.array(out, dtype=np.float64)


def _ikey(r: pd.Series) -> str:
    k = _row_kw(r)
    return f"{k['model_name']}|{k['method']}|{k['gpu_model']}|{k['num_gpus']}"


def _build_interaction(rows: pd.DataFrame, base_cal: dict) -> dict:
    """Return the per-config median residual (interaction_scale), rounded to 4 dp, for all models in ``rows``.

    ``base_cal`` holds the Tier 1 fit with an empty interaction_scale, so each ratio is what physics and
    Tier 1 leave unexplained for that cell."""
    pred = _predict(rows, base_cal)
    y = pd.to_numeric(rows["dataset_tokens_per_second"], errors="coerce").to_numpy(np.float64)
    df = pd.DataFrame({"key": [_ikey(r) for _, r in rows.iterrows()], "ratio": y / pred})
    df = df[np.isfinite(df["ratio"]) & (df["ratio"] > 0)]
    return {k: round(float(g["ratio"].median()), 4) for k, g in df.groupby("key", sort=True)}


def _fit_mgc_high(raw_path: Path, frozen_cal: dict, counts: tuple[int, ...] = (32, 128)) -> dict:
    """Fit mgc at the multi-node counts by median ratio, with every other scale frozen at the <=8 fit.

    A joint refit at these counts is non-identifiable. For each n, mgc[n] = median over resolvable
    n-GPU rows of prediction(mgc[n]=1) / measured. Returns {str(n): value} for each count with
    resolvable rows."""
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
    """Assemble multi_gpu_correction when every GPU count came from the joint Tier 1 fit (calibrate() path).

    ``mgc_fitted`` is Tier 1's by_num_gpus: the template's keys, fitted where Powell varied them and 1.0
    elsewhere. ``fit_counts`` are the counts Powell varied. A fitted count keeps its value. One GPU is a
    fixed anchor at 1.0, the value get_multi_gpu_correction returns for it. A gap gets log2-geometric
    interpolation of the nearest anchors below and above it (as the <=8 path does for 16/64), or the
    value of the highest anchor below it when none is above. With nothing fit, every count is 1.0."""
    keys = sorted(int(k) for k in mgc_fitted)
    anchors = {1: 1.0, **{c: mgc_fitted[str(c)] for c in fit_counts}}
    fit = sorted(anchors)
    out: dict[str, float] = {}
    for k in keys:
        if k in anchors:
            out[str(k)] = anchors[k]  # unrounded, like 2/4/8 on the <=8 path
            continue
        lower = [c for c in fit if c < k]
        upper = [c for c in fit if c > k]
        if lower and upper:
            lo, hi = lower[-1], upper[0]
            w = (np.log2(k) - np.log2(lo)) / (np.log2(hi) - np.log2(lo))
            val = float(np.exp((1 - w) * np.log(anchors[lo]) + w * np.log(anchors[hi])))
            out[str(k)] = round(val, 2)
        elif lower:
            out[str(k)] = round(float(anchors[lower[-1]]), 2)  # above the top fitted count: hold flat
        else:
            out[str(k)] = round(float(anchors[upper[0]]), 2)  # fewer than one GPU is not a real count: hold flat
    return out


def _fit_multi_gpu_correction(
    tier1: dict,
    rows: pd.DataFrame,
    train: pd.DataFrame,
    neutral: dict,
    interaction: dict,
    raw_path: Path | None,
    log: Callable[[str], None],
) -> tuple[dict, str]:
    """Return (mgc, mgc_note), the multi_gpu_correction table and its note (step 5 of the fit).

    The source depends on whether >8-GPU rows are in the main fit. The two sources are mutually
    exclusive, so no >8 row is counted twice."""
    mgc_lo = tier1["multi_gpu_correction"]["by_num_gpus"]
    high_totals = sorted({int(t) for t in rows["total"].unique() if t > 8})
    if high_totals:
        # Uncapped calibrate() path: the >8-GPU rows were in the joint Tier 1 fit, so mgc for every present
        # count comes from it. _fit_mgc_high would double-count those rows and divide out a comm_scale they
        # helped fit. Template counts missing from the data (e.g. 16/64) are interpolated.
        fitted_counts = {int(key) for (kind, key) in _vary_layout(train, neutral) if kind == "mgc"}
        mgc = _mgc_with_interpolated_gaps(mgc_lo, fitted_counts)
        mgc_note = (
            "Every GPU count present in the data was calibrated jointly by the Tier-1 Powell fit: the "
            ">8-GPU rows are part of the main fit, with no separate raw-trace step, so no >8 row is "
            f"double-counted. Counts fit directly: {sorted(fitted_counts)}. Template counts absent from "
            "the data are log2-geometric interpolations of the fitted neighbours (as the <=8 path does "
            "for 16/64). interaction_scale now covers whatever >8 cells the data provides."
        )
        log(f"  mgc: joint Tier-1 fit on GPU counts {sorted(fitted_counts)} (>8 in the main fit); gaps interpolated")
    elif raw_path is not None and raw_path.exists():
        frozen = copy.deepcopy(tier1)
        frozen["interaction_scale"] = interaction  # no keys above 8 GPUs, so no effect on this fit
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
            "2/4/8: joint Powell fit on the <=8-GPU profiling rows. 16/32/64/128 left neutral (1.0): "
            "the raw multi-node trace was unavailable at regeneration time, so >8 GPU is uncalibrated; "
            "the recommender restricts to <=8."
        )
        log(f"  WARNING: raw multi-node trace not found at {raw_path}; mgc >8 left neutral (1.0)")
    return mgc, mgc_note


def _fit_calibration(
    reference: dict,
    rows: pd.DataFrame,
    raw_path: Path | None,
    models: list[str] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[dict, dict]:
    """Fit the two-tier calibration from scratch on already-loaded valid ``rows``.

    Shared by regenerate() (internal CSV, capped at <=8 GPUs) and calibrate() (any CSV or DataFrame, no
    cap), so equivalent inputs give the same numbers. Step 5 picks one of two sources for the multi-GPU
    correction. Returns (calibration_dict, metrics); metrics holds row counts and held-out MdAPE."""
    # Step 1: raw kavier, every correction at 1.0, model_scale restricted to the selected set.
    neutral = _neutral_base(reference, models)

    # Step 2: seed-42 train/val/test split.
    train, val, test = train_val_test_split(rows)
    n_models = rows["model_name"].astype(str).nunique()
    max_total = int(rows["total"].max()) if len(rows) else 0
    log(
        f"  rows: train={len(train)} val={len(val)} test={len(test)} "
        f"(valid, tput>0, <={max_total} GPU, {n_models} models)"
    )

    # Step 3: Tier 1, regularized Powell fit of the global scales from the neutral prior; lambda picked
    #         on val, skipping non-physical candidates (_is_physical).
    tier1, info = select_calibration(
        train, val, grad_accum_steps=1, backward_factor=2.0, base_cal=neutral, accept=_is_physical
    )
    log(
        f"  Tier-1 fit: regularisation choice={info['choice']!r} val_mdape={info['val_mdape']:.2f}% "
        f"(comm_scale={tier1['comm_scale']:.3f})"
    )

    # Step 4: Tier 2, per-cell median(measured / pred) on train+val with Tier 1 applied and no interaction.
    base = copy.deepcopy(tier1)
    base["interaction_scale"] = {}
    train_val = pd.concat([train, val], ignore_index=True)
    interaction = _build_interaction(train_val, base)

    # Step 5: multi-GPU correction (see _fit_multi_gpu_correction).
    mgc, mgc_note = _fit_multi_gpu_correction(tier1, rows, train, neutral, interaction, raw_path, log)

    # Step 6: assemble in the shipped key order. Physics constants and schema/version come from the reference.
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

    # Held-out accuracy; the test split is not used by the fit.
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
    """Rebuild the whole table from scratch.

    ``models`` selects the model set (the profiling trace is filtered to it); None keeps every model,
    which reproduces calibration.json. Rows are capped at <=8 GPUs, so the >8-GPU mgc comes from the raw
    multi-node trace: mgc is one global communication-scaling scalar per GPU count, and the dense-4
    models have no multi-node rows. The cap keeps regeneration byte-identical; calibrate() has no cap."""
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
# Minimum valid rows for auto-selection by calibrate(models=None). The smallest shipped model,
# granite-3-8b, has 46. An explicit models= list bypasses this floor.
_MIN_ROWS_TO_AUTOSELECT = 8

# Minimum valid rows to fit a requested model: the 3-way split needs non-empty train/val/test. Thinner
# models are skipped with a warning. MIN_ROWS_PER_MODEL (30) only warns about thin but fittable models.
_MIN_ROWS_TO_FIT = 3


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _select_models(valid_rows: pd.DataFrame, min_rows: int = _MIN_ROWS_TO_AUTOSELECT) -> list[str]:
    """Return the sorted models in ``valid_rows`` with at least ``min_rows`` rows (calibrate()'s auto-set)."""
    counts = valid_rows["model_name"].astype(str).value_counts()
    return sorted(str(m) for m, c in counts.items() if int(c) >= min_rows)


def _suitability_report(valid: pd.DataFrame, models: list[str]) -> str | None:
    """Return one multi-line warning listing the failed suitability checks for ``models``, or None.

    The warning names the offending models and (model, GPU) cells. It is advisory: an unsuitable dataset
    is still fit. This is the headline warning of `kavier calibrate`."""
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


def _resolve_fit_models(valid: pd.DataFrame, models: list[str] | None) -> list[str]:
    """Return the models calibrate() fits.

    Starts from ``models`` (or the auto-selected set when None), drops models too thin for the 3-way
    split with a warning, and emits the suitability report once. Raises SystemExit if nothing is left.
    The suitability check does not change the result."""
    models_final = list(models) if models is not None else _select_models(valid)
    if not models_final:
        raise SystemExit(
            f"no model has >= {_MIN_ROWS_TO_AUTOSELECT} valid rows to auto-fit; pass models= to fit a specific set"
        )

    # A requested model below the split floor is skipped with a warning; the rest still fit.
    # Auto-selected models already clear _MIN_ROWS_TO_AUTOSELECT.
    counts_all = valid["model_name"].astype(str).value_counts()
    fittable = [m for m in models_final if int(counts_all.get(m, 0)) >= _MIN_ROWS_TO_FIT]
    skipped = [m for m in models_final if m not in fittable]
    if skipped:
        warnings.warn(
            f"skipping model(s) with fewer than {_MIN_ROWS_TO_FIT} valid rows (too few to fit): "
            + ", ".join(f"{m} (n={int(counts_all.get(m, 0))})" for m in skipped),
            UserWarning,
            stacklevel=3,
        )
    models_final = fittable
    if not models_final:
        raise SystemExit(f"no requested model has >= {_MIN_ROWS_TO_FIT} valid rows to fit (all skipped as too thin)")

    # One suitability warning naming the failed checks, or one info line when the data looks fine.
    report = _suitability_report(valid, models_final)
    if report is not None:
        warnings.warn(report, UserWarning, stacklevel=3)
    else:
        _eprint("dataset looks suitable")
    return models_final


def calibrate(source: str | Path | pd.DataFrame, models: list[str] | None = None) -> dict:
    """Fit a calibration table from scratch on ``source``, a profiling CSV path or a DataFrame.

    Uses the two-tier recipe of regenerate() with no total-GPU cap: >8-GPU rows join the main joint fit
    (see _fit_calibration step 5), so ``calibrate(profiling_trace.csv)`` does not reproduce
    calibration.json byte-for-byte. Keeps rows with ``is_valid == 1`` and
    ``dataset_tokens_per_second > 0``. For a path source, a sibling ``raw_trace.csv`` supplies the
    >8-GPU mgc only when the data has no >8 rows.

    ``models`` restricts the fit; None auto-selects every model with at least
    ``_MIN_ROWS_TO_AUTOSELECT`` valid rows. A missing REQUIRED_COLUMNS column is the only hard failure.
    A requested model with fewer than ``_MIN_ROWS_TO_FIT`` rows is skipped with a warning, and an
    unsuitable dataset (see _suitability_report) draws one advisory warning and is still fit.

    The shipped calibration.json is a structural template only (GPUs, methods, raw-physics constants,
    schema/version). No fitted scale is carried; the fit starts from a neutral 1.0 prior, seeded for any
    model, GPU or method the template lacks. Returns the calibration dict; ``_dumps`` serializes it in
    the shipped format."""
    reference = json.loads(CAL_PATH.read_text(encoding="utf-8"))

    if isinstance(source, pd.DataFrame):
        trace = source
        raw_path: Path | None = None
    else:
        src = Path(source)
        trace = pd.read_csv(src, low_memory=False)
        sibling = src.parent / RAW_MULTINODE_CSV
        raw_path = sibling if sibling.exists() else None

    # Uncapped: every valid row at any GPU count; ValueError if a required column is missing.
    valid = _filter_valid_rows(trace, None, max_total_gpus=None)
    if valid.empty:
        raise SystemExit("no valid rows (is_valid==1, dataset_tokens_per_second>0) in the input")

    models_final = _resolve_fit_models(valid, models)

    rows = _filter_valid_rows(trace, models_final, max_total_gpus=None)
    if rows.empty:
        raise SystemExit(f"no valid rows for models={models_final} in the input")

    # Seed 1.0 priors for models, GPUs and methods in the data but missing from the template, so a new GPU
    # or method is fit from its own rows (_get would raise KeyError). regenerate() does not run this.
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
    """Serialize like the shipped file (indent=2, trailing newline)."""
    return json.dumps(cal_dict, indent=2) + "\n"


# ======================================== CLI ========================================
def _resolve_models(model_set: str, models_csv: str | None) -> list[str]:
    """Return the models to fit: --models if given, else the named --model-set."""
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
    """Write the set's versions/ file; the 6-model set also overwrites the root calibration.json default."""
    VERSIONS_DIR.mkdir(exist_ok=True)
    target = VERSION_FILES[model_set]
    target.write_text(text)
    print(f"  wrote {target.relative_to(REPO_ROOT)}")
    if model_set == "6":
        CAL_PATH.write_text(text)
        print(f"  wrote {CAL_PATH.relative_to(REPO_ROOT)} (the default = 6-model)")


def _used_values(cal: dict) -> dict:
    """Return ``cal`` without its >8-GPU multi_gpu_correction entries and their note.

    Those entries are a median over every catalog model's large-GPU runs, so they move when models are
    added to the catalog; the recommender uses only <=8 GPUs."""
    out = json.loads(json.dumps(cal))
    mgc = out["multi_gpu_correction"]
    mgc["by_num_gpus"] = {k: v for k, v in mgc["by_num_gpus"].items() if int(k) <= 8}
    mgc.pop("_note", None)
    return out


def _check_both(reference: dict, args: argparse.Namespace) -> None:
    """Rebuild both model sets and check that they reproduce every used value of their files.

    Each set is compared with its versions/ file (see _used_values), and calibration.json with the
    6-model file byte for byte."""
    print(f"--check: rebuilding both model sets from {args.profiling_data} (template: {args.reference})")
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
        used_equal = _used_values(json.loads(new_text)) == _used_values(json.loads(target_text))
        print(f"  reproduces the used values of {target.name}: {used_equal}")
        print(f"  byte-for-byte, including >8-GPU mgc: {new_text == target_text}")
        if not used_equal:
            ok = False
            _print_diff(target_text, new_text, fromfile=target.name, tofile="regenerated")

    root_eq_6 = CAL_PATH.read_text(encoding="utf-8") == VERSION_FILES["6"].read_text(encoding="utf-8")
    print(f"\ncalibration.json == versions/{VERSION_FILES['6'].name}: {root_eq_6}")
    ok = ok and root_eq_6

    if not ok:
        raise SystemExit(
            "CHECK FAILED: a model set did not reproduce the used values of its versions/ file "
            "(re-run --write if the fit legitimately changed, else investigate non-determinism)"
        )
    print("\nCHECK PASSED: both model sets reproduce the used values of their versions/ files")


def _cmd_regen(reference: dict, args: argparse.Namespace) -> None:
    """Regenerate the selected --model-set and handle --write, --out and --snapshot."""
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

    # Report byte-identity against the set's versions/ file, with a diff on mismatch.
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
        help="calibration.json used as the structural template only (GPUs, methods, models, GPU counts), "
        "plus the raw-physics constants (mfu_batch_scale, training_overhead_s) and schema/version; no "
        "fitted scale is carried (default: the committed file)",
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
        "--models",
        dest="regen_models",
        help="comma-separated model list overriding --model-set (advanced); save the fit with --out",
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
        help="rebuild both model sets and check that each reproduces the used values of its versions/ file "
        "(all but the >8-GPU mgc) and that calibration.json equals the 6-model file; exit 1 if not",
    )

    args = ap.parse_args()
    if args.regen_models and args.write:
        ap.error("--write replaces the shipped 6-model or 4-model table; save a --models fit with --out PATH")

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if args.check:
        _check_both(reference, args)
        return
    _cmd_regen(reference, args)


if __name__ == "__main__":
    main()
