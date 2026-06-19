"""``kavier-co2`` CLI: estimate CO2 by billing fragments (training sim or OpenDC powerSource) against a carbon trace."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

import pandas as pd

from kavier_co2.emissions import EmissionResult, Fragment, compute_emissions, load_carbon_trace
from kavier_co2.fragments import fragments_from_powersource, fragments_from_training
from kavier_io.config import apply_config_defaults
from kavier_library.lookup import UnknownSpecError

_EXAMPLE_CMD = (
    "kavier-co2 --from-training --carbon_trace ct1-2025-ie-carbon-intensity.parquet "
    "--model_name mistral-7b-v0.1 --method lora --gpu_model NVIDIA-A100-SXM4-80GB "
    "--tokens_per_sample 1024 --batch_size 4 --number_gpus 8 --number_nodes 1 "
    "--total_tokens 100000000 --start_time '2025-06-01 00:00'"
)


class _FriendlyParser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        print(f"\nTry this example instead:\n  {_EXAMPLE_CMD}", file=sys.stderr)
        sys.exit(2)


def _build_parser() -> argparse.ArgumentParser:
    p = _FriendlyParser(description="Kavier CO2 emissions estimator", epilog=f"Example: {_EXAMPLE_CMD}")
    p.add_argument(
        "--config",
        default=None,
        help="YAML file of {arg_name: value} applied as defaults (explicit flags still override).",
    )
    p.add_argument("--carbon_trace", required=True, help="Path to carbon-intensity parquet (gCO2/kWh)")
    p.add_argument("--carbon_step_minutes", type=int, default=None, help="Override inferred trace step")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-training", action="store_true", help="Build fragments from a training sim")
    mode.add_argument("--powersource", default=None, help="Path to OpenDC powerSource.parquet")

    # --from-training args (mirror kavier-train).
    p.add_argument("--model_name")
    p.add_argument("--method", choices=["full", "lora", "gptq-lora"])
    p.add_argument("--gpu_model")
    p.add_argument("--tokens_per_sample", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--number_gpus", type=int)
    p.add_argument("--number_nodes", type=int)
    p.add_argument(
        "--total_tokens",
        type=int,
        default=None,
        help="Total tokens to train over (sets runtime); or use --epochs + --dataset_tokens.",
    )
    p.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Passes over the dataset; with --dataset_tokens derives total_tokens.",
    )
    p.add_argument(
        "--dataset_tokens", type=int, default=None, help="Tokens in one epoch of the dataset (used with --epochs)."
    )
    p.add_argument("--start_time", help="Run start (naive timestamp, e.g. '2025-06-01 00:00')")

    p.add_argument("--output_csv", default=None, help="Write the per-window breakdown to this CSV")
    return p


def _fragments_from_training_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[Fragment]:
    required = (
        "model_name",
        "method",
        "gpu_model",
        "tokens_per_sample",
        "batch_size",
        "number_gpus",
        "number_nodes",
        "start_time",
    )
    missing = [f"--{a}" for a in required if getattr(args, a) is None]
    # Job size may come from --total_tokens OR --epochs + --dataset_tokens (resolved by the engine).
    if args.total_tokens is None and not (args.epochs is not None and args.dataset_tokens is not None):
        missing.append("--total_tokens (or --epochs + --dataset_tokens)")
    if missing:
        parser.error(f"--from-training requires: {', '.join(missing)}")
    return fragments_from_training(
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
        start_time=pd.Timestamp(args.start_time),
    )


def _print_result(result: EmissionResult) -> None:
    print("=" * 60)
    print("Kavier CO2 Emissions")
    print("=" * 60)
    print(f"Total energy:        {result.total_energy_kwh:,.4f} kWh")
    print(f"Total CO2:           {result.total_co2_g:,.2f} g  ({result.total_co2_kg:,.4f} kg)")
    print(f"Avg intensity used:  {result.average_intensity:,.2f} gCO2/kWh (energy-weighted)")
    print(f"Windows touched:     {len(result.breakdown)}")
    print("=" * 60)


def _write_csv(result: EmissionResult, path: str) -> None:
    pd.DataFrame(result.breakdown).to_csv(path, index=False)
    print(f"Per-window breakdown written to {path}")


def _peek_config(argv: Optional[Sequence[str]]) -> Optional[str]:
    """Return the value of ``--config`` from ``argv`` (or ``sys.argv``), or ``None`` if absent."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", default=None)
    known, _ = peek.parse_known_args(argv)
    config: Optional[str] = known.config
    return config


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    # If --config is given, fold its YAML values in as defaults *before* parsing so any
    # explicit flag still overrides them; without it, behaviour is unchanged.
    config_path = _peek_config(argv)
    if config_path is not None:
        apply_config_defaults(parser, config_path)
    args = parser.parse_args(argv)

    trace = load_carbon_trace(args.carbon_trace, step_minutes=args.carbon_step_minutes)

    if args.powersource:
        ps = pd.read_parquet(args.powersource)
        fragments = fragments_from_powersource(ps)
    else:
        try:
            fragments = _fragments_from_training_args(args, parser)
        except UnknownSpecError as exc:
            print(f"{parser.prog}: error: {exc}", file=sys.stderr)
            sys.exit(2)

    try:
        result = compute_emissions(fragments, trace)
    except ValueError as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        sys.exit(2)

    _print_result(result)
    if args.output_csv:
        _write_csv(result, args.output_csv)


if __name__ == "__main__":
    main()
