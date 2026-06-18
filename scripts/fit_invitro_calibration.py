#!/usr/bin/env python
"""Fit per-model calibration (model_scale + interaction_scale) for a trace's models
and write it into the shipped calibration.json.

Method — Tier-2 median-ratio (the same convention the interaction_scale table uses):

    model_scale[m]                 = median( measured / physics_pred )      over the fit split
    interaction[m|method|gpu|n]    = median( measured / pred_with_model_scale ) per config cell

where physics_pred is the engine's analytical throughput with this model's own
model_scale/interaction removed (i.e. the global comm/mfu/mgc/method scales applied,
this model neutral at 1.0). The model_scale corrects the overall level; each
interaction cell corrects the residual for one (method, gpu, gpu-count) config.

Only <=8-GPU single-node rows are fit (the calibrated regime); >8-GPU configs keep
the global multi_gpu_correction extrapolation, matching the rest of the table. The
fit uses an 85% split and reports MdAPE on the held-out 15% (stratified by
method|gpu-count, deterministic seed). This is the routine used to add the
rwt3-llmbuild in-vitro models (granite-3.1-2b, granite-3.1-8b-instruct).

The global scales (comm_scale, mfu_multiplier, multi_gpu_correction, method_scale)
and the original dense-4 model_scales are fit separately by the Powell joint fitter
in coastline/benchmark/kavier_calibration.py; this script only adds/refreshes
individual per-model entries on top of those.

Data provenance: the profiling trace is IBM-internal (PD1 / ado-sfttrainer) and is
NOT vendored into kavier — pass its path with --trace. The trace must have columns:
model_name, method, gpu_model, number_gpus, number_nodes, batch_size,
tokens_per_sample, is_valid, dataset_tokens_per_second.

Usage:
    python scripts/fit_invitro_calibration.py \
        --trace ../trace-archive/pd1-profiling-dataset/ado-sfttrainer-for_invitro.csv \
        --models granite-3.1-2b granite-3.1-8b-instruct --write

Dry-run (print, don't touch calibration.json) by omitting --write.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))

import kavier_training.core.calibration as cal  # noqa: E402
from kavier_training.core.engine import simulate_training_step  # noqa: E402

CAL_PATH = _SRC / "kavier_training" / "data" / "calibration.json"
DEFAULT_TRACE = _HERE.parent.parent / "trace-archive" / "pd1-profiling-dataset" / "ado-sfttrainer-for_invitro.csv"
SEED = 42
TEST_FRAC = 0.15
MAX_GPUS = 8  # fit only the single-node calibrated regime; >8 uses global mgc


def stratified_split(df: pd.DataFrame, key_col: str, test_frac: float, seed: int):
    """Deterministic per-cell split: shuffle each stratum with a seeded RNG, hold out
    `test_frac` (>=1 row) for test. Reproduces exactly given the same trace + seed."""
    rng = np.random.default_rng(seed)
    test_idx: list[int] = []
    for _, g in df.groupby(key_col, sort=True):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx.extend(idx[:n_test].tolist())
    mask = df.index.isin(test_idx)
    return df[~mask], df[mask]


def _predict(rows: pd.DataFrame, model: str, cal_dict: dict) -> np.ndarray:
    saved = cal._CAL
    cal._CAL = cal_dict
    try:
        out = [
            simulate_training_step(
                model_name=model,
                gpu_model=str(r.gpu_model),
                tokens_per_sample=int(r.tokens_per_sample),
                batch_size=int(r.batch_size),
                method=str(r.method),
                num_gpus=int(r.total),
                num_nodes=int(r.number_nodes),
            )["tokens_per_second"]
            for r in rows.itertuples()
        ]
    finally:
        cal._CAL = saved
    return np.asarray(out, dtype=float)


def _mdape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.median(np.abs(p - y) / y) * 100.0)


def fit_model(model: str, trace: pd.DataFrame, base_cal: dict) -> dict:
    s = trace[
        (trace["model_name"] == model)
        & (trace["is_valid"] == 1.0)
        & (trace["dataset_tokens_per_second"] > 0)
    ].copy()
    s["total"] = (pd.to_numeric(s["number_gpus"]) * pd.to_numeric(s["number_nodes"])).astype(int)
    s = s[s["total"] <= MAX_GPUS].copy()
    if s.empty:
        raise SystemExit(f"no <= {MAX_GPUS}-GPU valid rows for {model!r} in the trace")
    s["cell"] = s["method"].astype(str) + "|" + s["total"].astype(str)
    s = s.reset_index(drop=True)
    train, test = stratified_split(s, "cell", TEST_FRAC, SEED)

    # base = global scales only; this model neutral (model_scale/interaction removed -> 1.0)
    base = copy.deepcopy(base_cal)
    base["model_scale"].pop(model, None)
    base["interaction_scale"] = {
        k: v for k, v in base["interaction_scale"].items() if not k.startswith(model + "|")
    }

    y_tr = train["dataset_tokens_per_second"].to_numpy(float)
    model_scale = float(np.median(y_tr / _predict(train, model, base)))

    with_ms = copy.deepcopy(base)
    with_ms["model_scale"][model] = model_scale
    interaction: dict[str, float] = {}
    for cell, g in train.groupby("cell", sort=True):
        method, ng = cell.split("|")
        key = f"{model}|{method}|{g['gpu_model'].iloc[0]}|{ng}"
        ratio = g["dataset_tokens_per_second"].to_numpy(float) / _predict(g, model, with_ms)
        interaction[key] = round(float(np.median(ratio)), 4)

    final = copy.deepcopy(with_ms)
    final["interaction_scale"].update(interaction)
    y_te = test["dataset_tokens_per_second"].to_numpy(float)
    held_out_mdape = round(_mdape(y_te, _predict(test, model, final)), 2)
    return {
        "model_scale": round(model_scale, 6),
        "interaction_scale": interaction,
        "n_fit": len(train),
        "n_test": len(test),
        "held_out_mdape": held_out_mdape,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, default=DEFAULT_TRACE, help="PD1 profiling CSV (IBM-internal; not vendored)")
    ap.add_argument("--models", nargs="+", default=["granite-3.1-2b", "granite-3.1-8b-instruct"])
    ap.add_argument("--write", action="store_true", help="write the result into calibration.json")
    args = ap.parse_args()

    if not args.trace.exists():
        raise SystemExit(f"trace not found: {args.trace}")
    trace = pd.read_csv(args.trace, low_memory=False)
    calj = json.loads(CAL_PATH.read_text())

    for model in args.models:
        res = fit_model(model, trace, cal._CAL)
        calj["model_scale"][model] = res["model_scale"]
        calj["interaction_scale"].update(res["interaction_scale"])
        print(
            f"{model}: model_scale={res['model_scale']}  cells={len(res['interaction_scale'])}  "
            f"fit_n={res['n_fit']} test_n={res['n_test']}  held-out MdAPE={res['held_out_mdape']}%"
        )

    if args.write:
        CAL_PATH.write_text(json.dumps(calj, indent=2) + "\n")
        print(f"\nwrote {CAL_PATH.relative_to(_HERE.parent)}")
    else:
        print("\n(dry-run; pass --write to update calibration.json)")


if __name__ == "__main__":
    main()
