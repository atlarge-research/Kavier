"""``kavier training`` subcommand: simulate a single config or every row of a CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence

from kavier.cli._args import add_training_args
from kavier.cli._shared import FriendlyParser, apply_config
from kavier.sdk.library.lookup import UnknownSpecError
from kavier.sdk.training.core.engine import simulate_full_training

_EXAMPLE_CMD = (
    "kavier training --model_name mistral-7b-v0.1 --method lora "
    "--gpu_model NVIDIA-A100-SXM4-80GB --tokens_per_sample 1024 "
    "--batch_size 4 --number_gpus 8 --number_nodes 1"
)


def _run_csv(path: str, total_tokens: int | None, epochs: float | None, dataset_tokens: int | None) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    header = (
        f"{'model':<28} {'method':<10} {'gpu':<22} {'seq':>5} {'bs':>3} {'gpus':>4} {'tok/s':>12} {'runtime_s':>10}"
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
            epochs=epochs,
            dataset_tokens=dataset_tokens,
        )
        print(
            f"{row['model_name']:<28} {row['method']:<10} {row['gpu_model']:<22} "
            f"{row['tokens_per_sample']:>5} {row['batch_size']:>3} {row['number_gpus']:>4} "
            f"{r['train_tokens_per_second']:>12,.1f} {r['train_runtime']:>10,.1f}"
        )
    print(f"\n{len(rows)} configurations simulated.")


def _require_single_config_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    single_cfg_args = (
        "model_name",
        "method",
        "gpu_model",
        "tokens_per_sample",
        "batch_size",
        "number_gpus",
        "number_nodes",
    )
    missing = [f"--{a}" for a in single_cfg_args if getattr(args, a) is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)} (or pass --input_csv)")


def _print_config_banner(args: argparse.Namespace, total_gpus: int) -> None:
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


def _run_single_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate the single-config flags, simulate one config, and print the banner + JSON result."""
    _require_single_config_args(parser, args)

    total_gpus = args.number_gpus * args.number_nodes
    _print_config_banner(args, total_gpus)

    results = simulate_full_training(
        model_name=args.model_name,
        method=args.method,
        gpu_model=args.gpu_model,
        tokens_per_sample=args.tokens_per_sample,
        batch_size=args.batch_size,
        number_gpus=args.number_gpus,
        number_nodes=args.number_nodes,
        total_tokens=args.total_tokens,
        epochs=args.epochs,
        dataset_tokens=args.dataset_tokens,
    )

    print("\nSimulation complete!")
    print(json.dumps(results, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    """Simulate a single config or all rows of a CSV."""
    parser = add_training_args(
        FriendlyParser(
            prog="kavier training",
            description="Kavier training simulator",
            epilog=f"Example: {_EXAMPLE_CMD}",
            example=_EXAMPLE_CMD,
        ),
    )
    # Fold --config YAML in as defaults BEFORE parsing, so explicit flags still override.
    apply_config(parser, argv)
    args = parser.parse_args(argv)

    if args.input_csv:
        try:
            _run_csv(args.input_csv, args.total_tokens, args.epochs, args.dataset_tokens)
        except UnknownSpecError as exc:
            parser.error(str(exc))
        return

    _run_single_config(parser, args)


if __name__ == "__main__":
    main()
