"""``kavier training`` subcommand: simulate a single config or every row of a CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence

from kavier.cli._args import add_training_args
from kavier.cli._shared import FriendlyParser, parse_args_with_config
from kavier.sdk.library.lookup import UnknownSpecError
from kavier.sdk.training.core.engine import normalise_method, simulate_full_training

_EXAMPLE_CMD = (
    "kavier training --model_name mistral-7b-v0.1 --method lora "
    "--gpu_model NVIDIA-A100-SXM4-80GB --tokens_per_sample 1024 "
    "--batch_size 4 --number_gpus 8 --number_nodes 1"
)


def _check_methods(path: str, rows: list[dict[str, str]]) -> None:
    """Raise ValueError giving the line of the first row whose method the engine does not accept."""
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        try:
            normalise_method(row["method"])
        except ValueError as exc:
            raise ValueError(f"{path} line {line}: {exc}") from None


def _run_csv(path: str, total_tokens: int | None, epochs: float | None, dataset_tokens: int | None) -> None:
    # utf-8-sig drops the byte-order mark Excel writes at the start of a UTF-8 CSV.
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    _check_methods(path, rows)
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
    """Simulate one config and print the banner and the JSON result."""
    _require_single_config_args(parser, args)
    if args.total_tokens is None and (args.epochs is None) != (args.dataset_tokens is None):
        parser.error("--epochs and --dataset_tokens go together (or use --total_tokens)")

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
    args = parse_args_with_config(parser, argv)

    try:
        if args.input_csv:
            _run_csv(args.input_csv, args.total_tokens, args.epochs, args.dataset_tokens)
        else:
            _run_single_config(parser, args)
    except (UnknownSpecError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
