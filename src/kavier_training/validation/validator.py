from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

from kavier_training.core.engine import simulate_full_training

_DATA = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_VALIDATION_CSV = _DATA / "input" / "validation_clean.csv"
_DEFAULT_OUTPUT_CSV = _DATA / "output" / "validation_results.csv"


def validate_predictions(
    validation_csv: str | Path | None = None,
    output_csv: str | Path | None = None,
    method: str = None,
    num_gpus: int = None,
    max_samples: int = None,
) -> Dict[str, Any]:
    validation_path = Path(validation_csv) if validation_csv else _DEFAULT_VALIDATION_CSV
    output_path = Path(output_csv) if output_csv else _DEFAULT_OUTPUT_CSV
    df = pd.read_csv(validation_path)

    if method:
        df = df[df["method"] == method]
    if num_gpus:
        df = df[df["number_gpus"] == num_gpus]
    if max_samples:
        df = df.head(max_samples)

    results = []
    errors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        try:
            pred = simulate_full_training(
                model_name=row["model_name"],
                method=row["method"],
                gpu_model=row["gpu_model"],
                tokens_per_sample=int(row["tokens_per_sample"]),
                batch_size=int(row["batch_size"]),
                number_gpus=int(row["number_gpus"]),
                number_nodes=int(row["number_nodes"]),
            )

            actual_throughput = row["measured_throughput"]
            predicted_throughput = pred["train_tokens_per_second"]
            error_pct = abs(predicted_throughput - actual_throughput) / actual_throughput * 100

            results.append(
                {
                    "model_name": row["model_name"],
                    "method": row["method"],
                    "gpu_model": row["gpu_model"],
                    "tokens_per_sample": row["tokens_per_sample"],
                    "batch_size": row["batch_size"],
                    "number_gpus": row["number_gpus"],
                    "number_nodes": row["number_nodes"],
                    "actual_throughput": actual_throughput,
                    "predicted_throughput": predicted_throughput,
                    "error_pct": error_pct,
                }
            )

        except Exception as e:
            errors.append({"index": idx, "model": row["model_name"], "error": str(e)})

    df_results = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)

    metrics = {
        "total_samples": len(df),
        "successful": len(results),
        "failed": len(errors),
        "mean_error_pct": df_results["error_pct"].mean(),
        "median_error_pct": df_results["error_pct"].median(),
        "min_error_pct": df_results["error_pct"].min(),
        "max_error_pct": df_results["error_pct"].max(),
        "std_error_pct": df_results["error_pct"].std(),
    }

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate Kavier training predictions")
    parser.add_argument("--method", type=str, help="Filter by method (full, lora, gptq-lora)")
    parser.add_argument("--num_gpus", type=int, help="Filter by number of GPUs")
    parser.add_argument("--max_samples", type=int, help="Limit number of samples")
    args = parser.parse_args()

    metrics = validate_predictions(
        method=args.method,
        num_gpus=args.num_gpus,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Successful: {metrics['successful']}")
    print(f"Failed: {metrics['failed']}")
    print("\nError Statistics:")
    print(f"  Mean: {metrics['mean_error_pct']:.2f}%")
    print(f"  Median: {metrics['median_error_pct']:.2f}%")
    print(f"  Std Dev: {metrics['std_error_pct']:.2f}%")
    print(f"  Min: {metrics['min_error_pct']:.2f}%")
    print(f"  Max: {metrics['max_error_pct']:.2f}%")
    print("=" * 60)
