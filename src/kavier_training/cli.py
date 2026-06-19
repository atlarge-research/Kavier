"""``kavier-train`` console entry point: simulate a single config or every row of a CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import NoReturn

from kavier_io.config import apply_config_defaults
from kavier_library.lookup import UnknownSpecError
from kavier_training.core.cli_args import add_training_args
from kavier_training.core.engine import simulate_full_training

_EXAMPLE_CMD = (
    "kavier-train --model_name mistral-7b-v0.1 --method lora "
    "--gpu_model NVIDIA-A100-SXM4-80GB --tokens_per_sample 1024 "
    "--batch_size 4 --number_gpus 8 --number_nodes 1"
)


class _FriendlyParser(argparse.ArgumentParser):
    """ArgumentParser whose error message appends a copy-pasteable example invocation."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        print(f"\nYou may have mistaken the input — try this example instead:\n  {_EXAMPLE_CMD}", file=sys.stderr)
        sys.exit(2)


def _peek_config(argv: list[str] | None = None) -> str | None:
    """Return the value of ``--config`` from ``argv`` (or ``sys.argv``), or ``None`` if absent."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", default=None)
    known, _ = peek.parse_known_args(argv)
    config: str | None = known.config
    return config


def _run_csv(path: str, total_tokens: int | None, epochs: float | None, dataset_tokens: int | None) -> None:
    """Simulate every row of a config CSV and print a throughput/runtime table."""
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


def main() -> None:
    parser = add_training_args(
        _FriendlyParser(
            description="Kavier training simulator",
            epilog=f"Example: {_EXAMPLE_CMD}",
        ),
    )
    # If --config is given, fold its YAML values in as defaults *before* parsing so any
    # explicit flag still overrides them; without it, behaviour is unchanged.
    config_path = _peek_config()
    if config_path is not None:
        apply_config_defaults(parser, config_path)
    args = parser.parse_args()

    if args.input_csv:
        try:
            _run_csv(args.input_csv, args.total_tokens, args.epochs, args.dataset_tokens)
        except UnknownSpecError as exc:
            parser.error(str(exc))
        return

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
        total_tokens=args.total_tokens,
        epochs=args.epochs,
        dataset_tokens=args.dataset_tokens,
    )

    print("\nSimulation complete!")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
