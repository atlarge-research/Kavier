"""Benchmark engine throughput accuracy across method/GPU-count/GPU-model slices and print a MAPE summary table."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from kavier_training.core.engine import simulate_training_step


def run_benchmark(df: pd.DataFrame, name: str, filter_fn=None) -> Dict:
    """Run the engine over an (optionally filtered) slice of ``df`` and return percent-error stats, or None if empty."""
    if filter_fn:
        subset = df[filter_fn(df)].copy()
    else:
        subset = df.copy()

    if len(subset) == 0:
        return None

    predicted = []
    actual = []
    errors = []

    for _, row in subset.iterrows():
        try:
            result = simulate_training_step(
                model_name=row["model_name"],
                gpu_model=row["gpu_model"],
                tokens_per_sample=int(row["tokens_per_sample"]),
                batch_size=int(row["batch_size"]),
                method=row["method"],
                num_gpus=int(row["number_gpus"]),
                num_nodes=int(row["number_nodes"]),
            )
            pred = result["tokens_per_second"]
            act = row["measured_throughput"]

            predicted.append(pred)
            actual.append(act)
            errors.append(abs(pred - act) / act * 100)
        except Exception as e:
            print(f"Error in {name}: {e}")
            continue

    if not predicted:
        return None

    return {
        "name": name,
        "samples": len(predicted),
        "mape": np.mean(errors),
        "median_ape": np.median(errors),
        "min_error": np.min(errors),
        "max_error": np.max(errors),
        "std_error": np.std(errors),
    }


def main():
    """Load the bundled validation CSV, run all benchmark slices, and print the MAPE table plus best/worst."""
    csv_path = Path(__file__).parent.parent / "data" / "input" / "validation_clean.csv"
    df = pd.read_csv(csv_path)

    print("=" * 70)
    print("KAVIER TRAINING SIMULATION BENCHMARKS")
    print("=" * 70)
    print(f"Total samples: {len(df)}\n")

    benchmarks = []

    # Overall benchmarks
    benchmarks.append(run_benchmark(df, "Overall (all samples)"))
    benchmarks.append(run_benchmark(df, "Full fine-tuning", lambda d: d["method"] == "full"))
    benchmarks.append(run_benchmark(df, "LoRA fine-tuning", lambda d: d["method"] == "lora"))

    # Single-GPU benchmarks
    benchmarks.append(run_benchmark(df, "Single-GPU (all)", lambda d: d["number_gpus"] == 1))
    benchmarks.append(run_benchmark(df, "Single-GPU full", lambda d: (d["number_gpus"] == 1) & (d["method"] == "full")))
    benchmarks.append(run_benchmark(df, "Single-GPU LoRA", lambda d: (d["number_gpus"] == 1) & (d["method"] == "lora")))

    # Multi-GPU benchmarks
    benchmarks.append(run_benchmark(df, "Multi-GPU (all)", lambda d: d["number_gpus"] > 1))
    benchmarks.append(run_benchmark(df, "2-GPU", lambda d: d["number_gpus"] == 2))
    benchmarks.append(run_benchmark(df, "4-GPU", lambda d: d["number_gpus"] == 4))
    benchmarks.append(run_benchmark(df, "8-GPU", lambda d: d["number_gpus"] == 8))

    # Per-GPU benchmarks
    for gpu in sorted(df["gpu_model"].unique()):
        benchmarks.append(
            run_benchmark(
                df,
                f"{gpu} (1-GPU)",
                lambda d, g=gpu: (d["gpu_model"] == g) & (d["number_gpus"] == 1),
            ),
        )
        benchmarks.append(
            run_benchmark(
                df,
                f"{gpu} (2-GPU)",
                lambda d, g=gpu: (d["gpu_model"] == g) & (d["number_gpus"] == 2),
            ),
        )

    # Print results
    print(f"{'Benchmark':<40} {'Samples':>8} {'MAPE':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)

    for b in benchmarks:
        if b:
            print(
                f"{b['name']:<40} {b['samples']:>8} {b['mape']:>7.1f}% "
                f"{b['median_ape']:>7.1f}% {b['min_error']:>7.1f}% {b['max_error']:>7.1f}%",
            )

    print("\n" + "=" * 70)

    # Summary of best/worst
    valid_benchmarks = [b for b in benchmarks if b]
    if valid_benchmarks:
        best = min(valid_benchmarks, key=lambda x: x["mape"])
        worst = max(valid_benchmarks, key=lambda x: x["mape"])

        print(f"Best:  {best['name']} (MAPE: {best['mape']:.1f}%)")
        print(f"Worst: {worst['name']} (MAPE: {worst['mape']:.1f}%)")

    print("=" * 70)


if __name__ == "__main__":
    main()
