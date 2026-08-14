#!/usr/bin/env python3
"""Plot CDF of per-job cross-trace ratios (patched / baseline) for JCT, wait, and runtime.

Ratios < 1 mean the patched trace is faster; > 1 means slower.
Jobs are matched by UUID (the 'min_gpu_recommender-' prefix is stripped from patched job IDs).

Wait ratio edge cases:
  - wait_b == 0 and wait_p == 0  →  ratio = 1.0  (no change)
  - wait_b == 0 and wait_p  > 0  →  denominator replaced with 1s (safe division),
                                     preserving the actual wait_p value as the ratio

Usage:
    python plot-job-cross-slowdown-cdf.py <baseline_per_jobs.csv> <patched_per_jobs.csv>

Output:
    job-cross-slowdown-cdf.pdf  (written to the current working directory)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

PREFIX = "min_gpu_recommender-"


def empirical_cdf(values):
    """Return (sorted_values, cumulative_probabilities) for an empirical CDF."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cdf = np.arange(1, n + 1) / n
    return sorted_vals, cdf


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline_per_jobs.csv> <patched_per_jobs.csv>")
        sys.exit(1)

    baseline_path, patched_path = sys.argv[1], sys.argv[2]

    baseline_df = pd.read_csv(baseline_path)
    patched_df = pd.read_csv(patched_path)

    # Normalise job IDs for merging: strip prefix and .yaml suffix to get bare UUID
    baseline_df["_uuid"] = baseline_df["job_id"].str.removeprefix(PREFIX).str.removesuffix(".yaml")
    patched_df["_uuid"] = patched_df["job_id"].str.removeprefix(PREFIX).str.removesuffix(".yaml")

    m = baseline_df.merge(patched_df, on="_uuid", suffixes=("_b", "_p"))

    jct_ratio = m["turnaround_s_p"] / m["turnaround_s_b"]

    # Safe wait ratio: replace zero denominator with 1 so nonzero/0 → wait_p (honest large ratio)
    wait_b_safe = m["wait_s_b"].where(m["wait_s_b"] > 0, 1.0)
    wait_ratio = m["wait_s_p"] / wait_b_safe

    runtime_ratio = m["runtime_s_p"] / m["runtime_s_b"]

    fig, ax = plt.subplots(figsize=(10, 4))

    for values, label, style in [
        (jct_ratio.values, "JCT ratio", "-"),
        (wait_ratio.values, "Wait ratio", "--"),
        (runtime_ratio.values, "Runtime ratio", ":"),
    ]:
        xs, ys = empirical_cdf(values)
        ax.plot(xs, ys, linestyle=style, label=label)

    ax.axvline(x=1, color="gray", linestyle="-.", linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel("Patched / Baseline Ratio", fontsize=16)
    ax.set_ylabel("CDF", fontsize=16)

    all_vals = np.concatenate([jct_ratio.values, wait_ratio.values, runtime_ratio.values])
    positive_vals = all_vals[all_vals > 0]
    ax.set_xlim(positive_vals.min(), positive_vals.max())
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig("job-cross-slowdown-cdf.pdf")


if __name__ == "__main__":
    main()
