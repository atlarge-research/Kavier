from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import NoReturn

from kavier_io.training_opendc import export_training_opendc
from kavier_training.core.cli_args import add_training_args
from kavier_training.core.engine import simulate_full_training, simulate_training_step
from library.lookup import UnknownSpecError

_EXAMPLE_CMD = (
    "kavier-train --model_name mistral-7b-v0.1 --method lora "
    "--gpu_model NVIDIA-A100-SXM4-80GB --tokens_per_sample 1024 "
    "--batch_size 4 --number_gpus 8 --number_nodes 1"
)


class _FriendlyParser(argparse.ArgumentParser):
    """On bad/missing arguments, show a ready-to-run example command."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        print(f"\nYou may have mistaken the input — try this example instead:\n  {_EXAMPLE_CMD}", file=sys.stderr)
        sys.exit(2)


def _run_csv(path: str, total_tokens: int | None) -> None:
    """Simulate every row of a CSV (e.g. data/input/input_example.csv)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    header = (
        f"{'model':<28} {'method':<10} {'gpu':<22} "
        f"{'seq':>5} {'bs':>3} {'gpus':>4} {'tok/s':>12} {'runtime_s':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        r = simulate_full_training(
            model_name=row["model_name"],
            method=row["method"],
            gpu_model=row["gpu_model"],
            tokens_per_sample=int(row["tokens_per_sample"]),
            batch_size=int(row["batch_size"]),
            number_gpus=int(row["number_gpus"]),
            number_nodes=int(row["number_nodes"]),
            total_tokens=total_tokens,
        )
        print(
            f"{row['model_name']:<28} {row['method']:<10} {row['gpu_model']:<22} "
            f"{row['tokens_per_sample']:>5} {row['batch_size']:>3} {row['number_gpus']:>4} "
            f"{r['train_tokens_per_second']:>12,.1f} {r['train_runtime']:>10,.1f}"
        )
    print(f"\n{len(rows)} configurations simulated.")


def main() -> None:
    parser = add_training_args(
        _FriendlyParser(
            description="Kavier training simulator",
            epilog=f"Example: {_EXAMPLE_CMD}",
        ),
    )
    args = parser.parse_args()

    if args.input_csv:
        try:
            _run_csv(args.input_csv, args.total_tokens)
        except UnknownSpecError as exc:
            parser.error(str(exc))
        return

    single_cfg_args = ("model_name", "method", "gpu_model", "tokens_per_sample",
                       "batch_size", "number_gpus", "number_nodes")
    missing = [f"--{a}" for a in single_cfg_args if getattr(args, a) is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)} (or pass --input_csv)")

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

    try:
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
    except UnknownSpecError as exc:
        parser.error(str(exc))

    print("\nSimulation complete!")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
