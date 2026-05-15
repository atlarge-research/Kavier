from __future__ import annotations

import argparse
import json

from kavier_training.core.cli_args import add_training_args
from kavier_training.core.engine import simulate_full_training


def main() -> None:
    parser = add_training_args(
        argparse.ArgumentParser(description="Kavier training simulator"),
    )
    args = parser.parse_args()
    total_gpus = args.number_gpus * args.number_nodes

    print("=" * 80)
    print("Kavier Training Simulator")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Method: {args.method}")
    print(f"GPU: {args.gpu_model}")
    print(f"Tokens per sample: {args.tokens_per_sample}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPUs: {args.number_gpus} x {args.number_nodes} nodes = {total_gpus} total")
    print("=" * 80)

    results = simulate_full_training(
        model_name=args.model_name,
        method=args.method,
        gpu_model=args.gpu_model,
        tokens_per_sample=args.tokens_per_sample,
        batch_size=args.batch_size,
        number_gpus=args.number_gpus,
        number_nodes=args.number_nodes,
    )

    print("\nSimulation complete!")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
