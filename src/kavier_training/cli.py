"""
Main entry point for Kavier training simulation.

This module provides the CLI command: kavier-train
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kavier_training.core.cli_args import add_training_args
from kavier_training.core.engine import simulate_full_training


def main() -> None:
    """
    Main entry point for kavier-train command.
    
    Parses CLI arguments and runs training simulation.
    """
    parser = argparse.ArgumentParser(
        description="Kavier Training Simulator - Predict LLM fine-tuning performance and energy"
    )
    parser = add_training_args(parser)
    args = parser.parse_args()
    
    print("=" * 80)
    print("Kavier Training Simulator")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Method: {args.method}")
    print(f"GPU: {args.gpu_model}")
    print(f"Tokens per sample: {args.tokens_per_sample}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPUs: {args.number_gpus} x {args.number_nodes} nodes = {args.number_gpus * args.number_nodes} total")
    print(f"Metrics: {args.metrics}")
    print("=" * 80)
    
    # Run simulation
    print("\nRunning simulation...")
    results = simulate_full_training(
        model_name=args.model_name,
        method=args.method,
        gpu_model=args.gpu_model,
        tokens_per_sample=args.tokens_per_sample,
        batch_size=args.batch_size,
        number_gpus=args.number_gpus,
        number_nodes=args.number_nodes,
        metrics=args.metrics,
    )
    
    # TODO: Save results to output folder
    # TODO: Print summary
    print("\nSimulation complete!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()

# Made with Bob
