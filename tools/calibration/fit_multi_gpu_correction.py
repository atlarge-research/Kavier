#!/usr/bin/env python
"""Fit the multi-GPU correction (`multi_gpu_correction.by_num_gpus`) for the MULTI-NODE
regime (>8 GPU) from measured ado-sfttrainer throughput, and optionally write it into the
shipped calibration.json.

METHOD — median-ratio (the same convention model_scale / interaction_scale use).
The engine computes training throughput proportional to `num_gpus / mgc[N]`, so for a
perfect fit `prediction_with_mgc[N]=1  /  mgc[N]  =  measured`, i.e.

    mgc[N] = median( prediction_with_mgc[N]_neutralised  /  measured )   over rows at N GPUs

Every OTHER calibration term (comm_scale, mfu_multiplier, method/model/interaction scales)
is left applied, so mgc[N] absorbs only the *residual* multi-node scaling those terms miss.

WHY THIS EXISTS. The single-node regime (<=8 GPU) is fit on the curated set elsewhere.
The shipped >8 values were interpolations (16, 64) or frozen raw-era anchors (32, 128)
never refit — which made predicted throughput non-monotonic in GPU count (8 GPUs beat
16/32/64). The raw ado-sfttrainer trace actually HAS multi-node measurements
(32 GPU ~ 630 valid rows, 128 ~ 571, 16 ~ 19 sparse; 64 has none), so 16/32/128 can be
fit directly from data. 64 has no data and is left as a log2 interpolation.

An 8-GPU row is always fit too, as a SANITY CHECK against the curated value (~1.15).

FINDING (first run on ado-sfttrainer-raw, 2026-06-18). The refit barely moves 32/128
(32: 4.89->4.96, 128: 17.79->17.42) -- the shipped "frozen anchors" already WERE the
median-ratio values for this data. BUT the 8-GPU sanity fit lands at ~3.59, far from the
curated dense-4 value 1.148: the correction is MODEL-DEPENDENT. Every >8-GPU row here is a
large/instruct model (llama3.1-70b, mixtral, granite-3.1-8b-instruct, granite-3.1-3b-a800m);
the dense-4 recommender models (mistral-7b, granite-3.3-8b, llama3.2-3b, granite-3-8b) have
ZERO >8 rows, and 16-GPU has zero library-resolvable rows at all. So this data CANNOT
calibrate the dense-4 >8 regime -- their >8 predictions stay extrapolations, which is exactly
why the recommender restricts to <=8 GPU. Honestly fixing >8 for the dense-4 needs dedicated
dense-4 multi-node benchmarks (e.g. the Zurich cluster). The refit therefore confirms the
existing values rather than changing them; this script is kept for provenance + reproducibility.

USAGE
    # dry-run (print only; includes the 8-GPU sanity check)
    python scripts/fit_multi_gpu_correction.py \
        --trace ../trace-archive/pd1-profiling-dataset/ado-sfttrainer-raw.csv
    # write the fitted >8 values into calibration.json
    python scripts/fit_multi_gpu_correction.py --trace <csv> --counts 16 32 128 --write

The trace is IBM-internal (ado-sfttrainer) and is NOT vendored into kavier; pass its path.
Required columns: model_name, gpu_model, method, number_gpus, number_nodes, batch_size,
tokens_per_sample, is_valid, dataset_tokens_per_second.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

# _common bootstraps sys.path (kavier src) so the kavier_* imports below resolve.
from _common import CAL_PATH, REPO_ROOT, TRACE_ARCHIVE, base_arg_parser, calibration_override

from kavier_library.gpu import GPU_SPEC_LIBRARY  # noqa: E402
from kavier_library.llm import LLM_SPEC_LIBRARY  # noqa: E402
from kavier_training.core.engine import simulate_training_step  # noqa: E402

DEFAULT_TRACE = TRACE_ARCHIVE / "ado-sfttrainer-raw.csv"


def fit_count(trace: pd.DataFrame, n: int, base_cal: dict) -> list[float]:
    """Return the per-row ratios prediction(mgc[n]=1)/measured for valid, resolvable rows
    at `n` total GPUs. mgc[n] is the median of these ratios."""
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


def main() -> None:
    ap = base_arg_parser(__doc__)
    ap.set_defaults(trace=DEFAULT_TRACE)
    ap.add_argument(
        "--counts", type=int, nargs="+", default=[16, 32, 128], help="total-GPU counts to fit (and write with --write)"
    )
    args = ap.parse_args()

    if not args.trace.exists():
        raise SystemExit(f"trace not found: {args.trace}")
    t = pd.read_csv(args.trace, low_memory=False)
    t = t[(t["is_valid"] == 1.0) & (t["dataset_tokens_per_second"] > 0)].copy()
    t["tot"] = (pd.to_numeric(t["number_gpus"]) * pd.to_numeric(t["number_nodes"])).astype(int)

    calj = json.loads(CAL_PATH.read_text())
    cur = calj["multi_gpu_correction"]["by_num_gpus"]

    print("median-ratio multi-GPU correction fit (eff = num_gpus / mgc; higher eff = better scaling)")
    print(f"{'GPUs':>5} {'n_rows':>7} {'mgc_old':>9} {'mgc_new':>9} {'eff_old':>8} {'eff_new':>8}")
    fitted: dict[int, float] = {}
    for n in sorted(set([8, *args.counts])):
        # cur is not mutated until after the loop, so calj == the shipped base here.
        ratios = fit_count(t, n, calj)
        if not ratios:
            print(f"{n:>5} {0:>7}   (no resolvable rows — skipped)")
            continue
        mgc_new = float(np.median(ratios))
        mgc_old = float(cur.get(str(n), float("nan")))
        eff_old = n / mgc_old if mgc_old == mgc_old else float("nan")
        print(f"{n:>5} {len(ratios):>7} {mgc_old:>9.3f} {mgc_new:>9.3f} {eff_old:>8.2f} {n / mgc_new:>8.2f}")
        fitted[n] = round(mgc_new, 4)

    if args.write:
        written = [n for n in args.counts if n in fitted]
        for n in written:
            cur[str(n)] = fitted[n]
        CAL_PATH.write_text(json.dumps(calj, indent=2) + "\n")
        print(f"\nwrote mgc {written} -> {CAL_PATH.relative_to(REPO_ROOT)}")
    else:
        print("\n(dry-run; pass --write to update calibration.json)")


if __name__ == "__main__":
    main()
