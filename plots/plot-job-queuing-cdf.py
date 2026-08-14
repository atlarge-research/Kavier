#!/usr/bin/env python3
"""Plot CDF of queuing times for two simulation traces.

Usage:
    python plot-job-queuing-cdf.py <baseline_per_jobs.csv> <patched_per_jobs.csv>

Output:
    job-queuing-cdf.pdf  (written to the current working directory)
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

    baseline = pd.read_csv(baseline_path)["wait_s"].dropna().values / 3600
    patched = pd.read_csv(patched_path)["wait_s"].dropna().values / 3600

    fig, ax = plt.subplots(figsize=(10, 4))

    for values, label, style in [
        (baseline, "Baseline", "-"),
        (patched, "Patched", "--"),
    ]:
        xs, ys = empirical_cdf(values)
        ax.plot(xs, ys, linestyle=style, label=label)

    ax.set_xlabel("Queuing Time (h)", fontsize=16)
    ax.set_ylabel("CDF", fontsize=16)

    ax.set_xlim(0, max(baseline.max(), patched.max()))
    ax.set_ylim(0.8, 1)
    ax.grid(True)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig("job-queuing-cdf.pdf")


if __name__ == "__main__":
    main()
