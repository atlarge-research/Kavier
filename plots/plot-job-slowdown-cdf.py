#!/usr/bin/env python3
"""Plot CDF of per-job scheduling slowdown (JCT / runtime) for two simulation traces.

A slowdown of 1.0 means the job ran immediately with no queueing overhead.
Higher values reflect scheduling delay.

Usage:
    python plot-job-slowdown-cdf.py <baseline_per_jobs.csv> <patched_per_jobs.csv>

Output:
    job-slowdown-cdf.pdf  (written to the current working directory)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


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

    baseline = (baseline_df["turnaround_s"] / baseline_df["runtime_s"]).dropna().values
    patched = (patched_df["turnaround_s"] / patched_df["runtime_s"]).dropna().values

    fig, ax = plt.subplots(figsize=(10, 4))

    for values, label, style in [
        (baseline, "Baseline", "-"),
        (patched, "Patched", "--"),
    ]:
        xs, ys = empirical_cdf(values)
        ax.plot(xs, ys, linestyle=style, label=label)

    ax.axvline(x=1, color="gray", linestyle=":", linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel("Scheduling Slowdown (JCT / Runtime)", fontsize=16)
    ax.set_ylabel("CDF", fontsize=16)

    all_vals = np.concatenate([baseline, patched])
    ax.set_xlim(all_vals.min(), all_vals.max())
    ax.set_ylim(0.8, 1)
    ax.grid(True)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig("job-slowdown-cdf.pdf")


if __name__ == "__main__":
    main()
