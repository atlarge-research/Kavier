from __future__ import annotations

import argparse
import json
import sys

from kavier_io.training_opendc import export_training_opendc
from kavier_training.core.cli_args import add_training_args
from kavier_training.core.engine import simulate_full_training, simulate_training_step

_EXAMPLE_CMD = (
    "kavier-train --model_name mistral-7b-v0.1 --method lora "
    "--gpu_model NVIDIA-A100-SXM4-80GB --tokens_per_sample 1024 "
    "--batch_size 4 --number_gpus 8 --number_nodes 1"
)


class _FriendlyParser(argparse.ArgumentParser):
    """On bad/missing arguments, show a ready-to-run example command."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        print(f"\nYou may have mistaken the input — try this example instead:\n  {_EXAMPLE_CMD}", file=sys.stderr)
        sys.exit(2)


def main() -> None:
    parser = add_training_args(
        _FriendlyParser(
            description="Kavier training simulator",
            epilog=f"Example: {_EXAMPLE_CMD}",
        ),
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

    if args.opendc_output_dir:
        results = export_training_opendc(
            output_dir=args.opendc_output_dir,
            model_name=args.model_name,
            method=args.method,
            gpu_model=args.gpu_model,
            tokens_per_sample=args.tokens_per_sample,
            batch_size=args.batch_size,
            number_gpus=args.number_gpus,
            number_nodes=args.number_nodes,
            total_tokens=args.total_tokens,
            task_id=args.opendc_task_id,
            submission_time_ms=args.opendc_submission_time_ms,
            simulate_full_training_fn=simulate_full_training,
            simulate_training_step_fn=simulate_training_step,
        )
    else:
        results = simulate_full_training(
            model_name=args.model_name,
            method=args.method,
            gpu_model=args.gpu_model,
            tokens_per_sample=args.tokens_per_sample,
            batch_size=args.batch_size,
            number_gpus=args.number_gpus,
            number_nodes=args.number_nodes,
            total_tokens=args.total_tokens,
        )

    print("\nSimulation complete!")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
