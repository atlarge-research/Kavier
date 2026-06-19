#!/usr/bin/env python
"""Fit Kavier's empirical (Tier-2) calibration on the 85% TRAIN+VAL split and
evaluate held-out throughput accuracy on the untouched 15% TEST split.

WHY. The shipped calibration.json fit its residual scales (model_scale,
interaction_scale, ...) on 100% of the curated data -- in-sample, so its ~5-6%
MdAPE is a leakage-inflated upper bound. For an honest comparison against the
ML predictors (which are trained on a 70/15/15 split and reported on the 15%
test), Kavier must be calibrated on the SAME 85% (train+val) and evaluated on
the SAME held-out 15% test. This script does exactly that and is kept for
provenance: it shows, end to end, how the calibration is produced.

METHOD -- greedy median-ratio decomposition (matches the engine's structure).
The engine assembles throughput as

    tps = [grad_accum * batch * tok/sample * num_gpus / mgc[N]] / step_time_s
          * (method_scale * model_scale * interaction_scale)

comm_scale and mfu_multiplier live INSIDE step_time_s / the MFU (the physics
baseline) and are LEFT at their shipped values -- they are global/per-GPU, low
leakage risk, and (importantly) leaving mfu_multiplier untouched keeps Kavier's
POWER prediction bit-identical, so the energy side of the experiments is
unaffected. We refit only the empirical residual that the engine applies as
pure post-multipliers on the physics throughput:

    pred_uncorrected = physics(comm,mfu) * num_gpus           (mgc=1, scales=1)
    r0 = measured / pred_uncorrected                          (total correction)

    mgc[N]        = median( pred_uncorrected / measured )      over rows at N GPUs (N>1)
                    (mgc DIVIDES, so it takes the reciprocal convention)
    method_scale  = median( measured / pred_after_mgc )        per method
    model_scale   = median( measured / pred_after_method )     per model
    interaction   = median( measured / pred_after_model )      per (model|method|gpu|N) key

Each scale is fit on the residual left by the coarser ones (mgc -> method ->
model -> interaction), exactly how the engine layers them. interaction_scale is
only fit for keys with >= MIN_INTERACTION_ROWS samples (single-row keys are
noise and never generalise); unseen keys fall back to a neutral 1.0 in the
engine, so on the 15% test the finest term is mostly neutral -- by design.

The split is the SAME deterministic 70/15/15 (SEED=42) used by every trainer in
coastline/trainer/common.py, reproduced through the identical load+split path so
the 15% test rows are byte-for-byte the ones the ML models are evaluated on.

USAGE
    # dry-run: fit + print the 85% in-sample and 15% held-out MdAPE, no write
    python scripts/fit_calibration.py
    # also write the fitted calibration JSON (used by the accuracy comparison)
    python scripts/fit_calibration.py --write
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
_KAVIER_SRC = _REPO / "kavier" / "src"
_COASTLINE = _REPO / "coastline"
for p in (str(_KAVIER_SRC), str(_COASTLINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_TRACE = _REPO / "trace-archive" / "pd1-profiling-dataset" / "ado-sfttrainer_curated_6models.csv"
DATA6_DIR = _COASTLINE / "trainer" / ".data6model"
CAL_PATH = _KAVIER_SRC / "kavier_training" / "data" / "calibration.json"
DEFAULT_OUT = _KAVIER_SRC / "kavier_training" / "data" / "calibration_6models_85.json"

MIN_INTERACTION_ROWS = 3

# Point the trainer's loader at the 6-model set BEFORE importing trainer.common
# (DATA_PATH is resolved at import time from DATA_DIR).
os.environ.setdefault("DATA_DIR", str(DATA6_DIR))

from trainer.common import (  # noqa: E402
    load_and_preprocess_data,
    split_data,
    transform_targets,
)

import kavier_training.core.calibration as cal  # noqa: E402
from kavier_training.core.engine import simulate_training_step  # noqa: E402


def _filter_like_loader(df: pd.DataFrame) -> pd.DataFrame:
    """Apply EXACTLY the row filter load_and_preprocess_data uses, so the index
    aligns with the trainer's X (which carries model_name etc. that X drops)."""
    if "is_valid" in df.columns:
        df = df.loc[df["is_valid"] == 1.0]
    mask = (
        df["dataset_tokens_per_second"].notna()
        & (df["dataset_tokens_per_second"] > 0)
        & df["train_runtime"].notna()
        & (df["train_runtime"] > 0)
    )
    return df.loc[mask].copy()


def _trainer_split_indices():
    """Reproduce the trainer's EXACT (X, y, y_log) split and return the
    train+val ('85%') and test ('15%') row indices."""
    X_cat, X_num, y, _, _ = load_and_preprocess_data()
    X = pd.concat([pd.DataFrame(X_cat).astype(str), pd.DataFrame(X_num)], axis=1)
    y_log = transform_targets(y)
    (X_tr, _, _), (X_val, _, _), (X_te, _, _) = split_data(X, y, y_log)
    trainval_idx = X_tr.index.append(X_val.index)
    return trainval_idx, X_te.index, X_cat.index


def _simulate_tps(row) -> float:
    tot = int(row["number_gpus"]) * int(row["number_nodes"])
    return simulate_training_step(
        model_name=str(row["model_name"]),
        gpu_model=str(row["gpu_model"]),
        tokens_per_sample=int(row["tokens_per_sample"]),
        batch_size=int(row["batch_size"]),
        method=str(row["method"]),
        num_gpus=tot,
        num_nodes=int(row["number_nodes"]),
    )["tokens_per_second"]


def _mdape(measured: np.ndarray, pred: np.ndarray) -> float:
    m = measured > 0
    return float(np.median(np.abs((pred[m] - measured[m]) / measured[m])) * 100.0)


def fit(df85: pd.DataFrame, shipped: dict) -> dict:
    """Greedy median-ratio fit of mgc / method / model / interaction on the 85%."""
    df = df85.copy()
    df["tot"] = (df["number_gpus"].astype(int) * df["number_nodes"].astype(int)).astype(int)

    # 1. pred_uncorrected: Tier-2 neutral, Tier-1 (comm/mfu) at shipped values.
    neutral = copy.deepcopy(shipped)
    neutral["method_scale"] = {}
    neutral["model_scale"] = {}
    neutral["interaction_scale"] = {}
    neutral["multi_gpu_correction"]["by_num_gpus"] = {str(n): 1.0 for n in sorted(df["tot"].unique()) if n > 1}
    saved = cal._CAL
    cal._CAL = neutral
    try:
        df["pred_unc"] = [_simulate_tps(r) for _, r in df.iterrows()]
    finally:
        cal._CAL = saved
    df = df[(df["pred_unc"] > 0) & (df["dataset_tokens_per_second"] > 0)].copy()
    meas = df["dataset_tokens_per_second"].to_numpy(dtype=float)

    # 2. mgc[N] (divides): median(pred_unc / measured) over rows at N>1 GPUs.
    mgc: dict[str, float] = {}
    pred = df["pred_unc"].to_numpy(dtype=float).copy()
    for n in sorted(df["tot"].unique()):
        if n <= 1:
            continue
        sel = (df["tot"] == n).to_numpy()
        mgc[str(n)] = float(np.median(pred[sel] / meas[sel]))
        pred[sel] = pred[sel] / mgc[str(n)]

    # 3. method_scale (multiplies): median(measured / pred_after_mgc) per method.
    method_scale: dict[str, float] = {}
    for meth in sorted(df["method"].unique()):
        sel = (df["method"] == meth).to_numpy()
        method_scale[str(meth)] = float(np.median(meas[sel] / pred[sel]))
        pred[sel] = pred[sel] * method_scale[str(meth)]

    # 4. model_scale (multiplies): median(measured / pred_after_method) per model.
    model_scale: dict[str, float] = {}
    for mdl in sorted(df["model_name"].unique()):
        sel = (df["model_name"] == mdl).to_numpy()
        model_scale[str(mdl)] = float(np.median(meas[sel] / pred[sel]))
        pred[sel] = pred[sel] * model_scale[str(mdl)]

    # 5. interaction_scale (multiplies): per (model|method|gpu|N), guarded by row count.
    interaction_scale: dict[str, float] = {}
    df = df.assign(_pred=pred)
    keys = (
        df["model_name"].astype(str)
        + "|"
        + df["method"].astype(str)
        + "|"
        + df["gpu_model"].astype(str)
        + "|"
        + df["tot"].astype(str)
    )
    df = df.assign(_key=keys)
    for key, grp in df.groupby("_key"):
        if len(grp) < MIN_INTERACTION_ROWS:
            continue
        ratio = grp["dataset_tokens_per_second"].to_numpy(float) / grp["_pred"].to_numpy(float)
        interaction_scale[key] = float(np.median(ratio))

    fitted = copy.deepcopy(shipped)
    # Keep shipped >8 mgc anchors; overwrite the fitted single-node counts.
    fitted["multi_gpu_correction"]["by_num_gpus"].update(mgc)
    fitted["method_scale"] = method_scale
    fitted["model_scale"] = model_scale
    fitted["interaction_scale"] = interaction_scale
    fitted["_provenance"] = {
        "fit": "Tier-2 (mgc/method/model/interaction) median-ratio on 85% train+val",
        "kept_from_shipped": ["comm_scale", "mfu_multiplier", "training_overhead_s", ">8 mgc anchors"],
        "min_interaction_rows": MIN_INTERACTION_ROWS,
    }
    return fitted


def eval_mdape(df: pd.DataFrame, fitted: dict, label: str) -> dict:
    saved = cal._CAL
    cal._CAL = fitted
    try:
        pred = np.array([_simulate_tps(r) for _, r in df.iterrows()], dtype=float)
    finally:
        cal._CAL = saved
    meas = df["dataset_tokens_per_second"].to_numpy(dtype=float)
    overall = _mdape(meas, pred)
    per_model = {}
    for mdl, grp in df.assign(_pred=pred).groupby("model_name"):
        per_model[str(mdl)] = (
            _mdape(grp["dataset_tokens_per_second"].to_numpy(float), grp["_pred"].to_numpy(float)),
            int(len(grp)),
        )
    print(f"\n{label}: Kavier throughput MdAPE = {overall:.2f}%  (n={len(df)})")
    print(f"  {'model':28s} {'MdAPE%':>8s} {'n':>5s}")
    for mdl in sorted(per_model):
        md, n = per_model[mdl]
        print(f"  {mdl:28s} {md:8.2f} {n:5d}")
    return {"overall": overall, "per_model": per_model}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--write", action="store_true", help="write the fitted calibration JSON")
    args = ap.parse_args()

    shipped = json.loads(CAL_PATH.read_text())

    df_full = _filter_like_loader(pd.read_csv(args.trace, low_memory=False))
    trainval_idx, test_idx, loader_idx = _trainer_split_indices()
    # Prove df_full and the trainer's X cover the identical row set.
    assert set(df_full.index) == set(loader_idx), (
        f"row-set mismatch: df_full={len(df_full)} loader={len(loader_idx)} "
        "(trace fed to the trainer differs from --trace)"
    )
    df85 = df_full.loc[trainval_idx]
    df15 = df_full.loc[test_idx]
    print(f"split: 85% train+val = {len(df85)} rows | 15% test = {len(df15)} rows (SEED=42)")

    fitted = fit(df85, shipped)
    print(
        "\nfitted multi_gpu_correction (<=8):",
        {
            k: round(v, 3)
            for k, v in sorted(fitted["multi_gpu_correction"]["by_num_gpus"].items(), key=lambda x: int(x[0]))
            if int(k) <= 8
        },
    )
    print("fitted method_scale:", {k: round(v, 3) for k, v in fitted["method_scale"].items()})
    print("fitted model_scale :", {k: round(v, 3) for k, v in fitted["model_scale"].items()})
    print(f"fitted interaction_scale: {len(fitted['interaction_scale'])} keys (>= {MIN_INTERACTION_ROWS} rows)")

    eval_mdape(df85, fitted, "IN-SAMPLE (85% train+val)")
    eval_mdape(df15, fitted, "HELD-OUT (15% test)")

    if args.write:
        args.out.write_text(json.dumps(fitted, indent=2) + "\n")
        print(f"\nwrote fitted calibration -> {args.out.relative_to(_REPO)}")
    else:
        print("\n(dry-run; pass --write to save the fitted calibration JSON)")


if __name__ == "__main__":
    main()
