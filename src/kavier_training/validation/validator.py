"""
Validation script for training predictions.

Compares Kavier predictions against real measurements from ado-sfttrainer dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from library.training_metrics import TRAINING_PERFORMANCE_METRICS, TRAINING_ENERGY_METRICS


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load validation dataset.
    
    Args:
        dataset_path: Path to ado-sfttrainer_valid_runs.csv
        
    Returns:
        DataFrame with validation data
    """
    df = pd.read_csv(dataset_path)
    return df


def calculate_mape(actual: float, predicted: float) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    MAPE = |actual - predicted| / actual * 100
    
    Args:
        actual: Actual measured value
        predicted: Predicted value
        
    Returns:
        MAPE percentage
    """
    if actual == 0:
        return 0.0
    return abs(actual - predicted) / actual * 100


def calculate_total_energy(power_watts: float, runtime_seconds: float) -> float:
    """
    Calculate total energy consumption.
    
    Energy (Wh) = Power (W) * Time (h)
    
    Args:
        power_watts: Average power consumption in watts
        runtime_seconds: Training runtime in seconds
        
    Returns:
        Total energy in Wh
    """
    runtime_hours = runtime_seconds / 3600.0
    return power_watts * runtime_hours


def validate_single_run(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    number_gpus: int,
    number_nodes: int,
    actual_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Validate a single training run.
    
    Args:
        model_name: LLM model name
        method: Fine-tuning method (full/lora)
        gpu_model: GPU model
        tokens_per_sample: Sequence length
        batch_size: Batch size per GPU
        number_gpus: Number of GPUs
        number_nodes: Number of nodes
        actual_metrics: Actual measured metrics from dataset
        
    Returns:
        Dictionary with predictions and errors
    """
    # TODO: Call simulator to get predictions
    # from simulator.training.simulate import simulate_full_training
    # predictions = simulate_full_training(...)
    
    # Placeholder predictions
    predictions = {
        "dataset_tokens_per_second": 0.0,
        "train_runtime": 0.0,
        "gpu_compute_utilization_avg": 0.0,
        "gpu_power_watts_avg": 0.0,
        "total_energy_wh": 0.0,
    }
    
    # Calculate errors for performance metrics
    errors = {}
    for metric in TRAINING_PERFORMANCE_METRICS:
        if metric in actual_metrics and metric in predictions:
            errors[f"{metric}_mape"] = calculate_mape(
                actual_metrics[metric], predictions[metric]
            )
    
    # Calculate errors for energy metrics
    for metric in TRAINING_ENERGY_METRICS:
        if metric in actual_metrics and metric in predictions:
            errors[f"{metric}_mape"] = calculate_mape(
                actual_metrics[metric], predictions[metric]
            )
    
    return {
        "predictions": predictions,
        "actual": actual_metrics,
        "errors": errors,
    }


def validate_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Validate all runs in dataset.
    
    Args:
        dataset_path: Path to validation dataset
        
    Returns:
        DataFrame with validation results
    """
    df = load_dataset(dataset_path)
    
    results = []
    for _, row in df.iterrows():
        # Calculate total energy from power and runtime
        total_energy_wh = calculate_total_energy(
            float(row["gpu_power_watts_avg"]),
            float(row["train_runtime"])
        )
        
        actual_metrics = {
            "dataset_tokens_per_second": float(row["dataset_tokens_per_second"]),
            "train_runtime": float(row["train_runtime"]),
            "gpu_compute_utilization_avg": float(row["gpu_compute_utilization_avg"]),
            "gpu_power_watts_avg": float(row["gpu_power_watts_avg"]),
            "total_energy_wh": total_energy_wh,
        }
        
        result = validate_single_run(
            model_name=str(row["model_name"]),
            method=str(row["method"]),
            gpu_model=str(row["gpu_model"]),
            tokens_per_sample=int(float(row["tokens_per_sample"])),
            batch_size=int(float(row["batch_size"])),
            number_gpus=int(float(row["number_gpus"])),
            number_nodes=int(float(row["number_nodes"])),
            actual_metrics=actual_metrics,
        )
        
        results.append({
            "identifier": row["identifier"],
            "model_name": row["model_name"],
            "method": row["method"],
            **result["predictions"],
            **result["errors"],
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    dataset_path = "finetuning-data/in/curated/ado-sfttrainer_valid_runs.csv"
    results_df = validate_dataset(dataset_path)
    
    print("Validation Results:")
    print(results_df.head())
    
    # Calculate average MAPE for each metric
    print("\n" + "="*60)
    print("Average MAPE by Metric:")
    print("="*60)
    
    mape_cols = [col for col in results_df.columns if col.endswith("_mape")]
    for col in mape_cols:
        avg_mape = results_df[col].mean()
        metric_name = col.replace("_mape", "")
        print(f"{metric_name:40s}: {avg_mape:6.2f}%")


