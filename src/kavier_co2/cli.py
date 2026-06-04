"""kavier-co2 — link energy consumption to CO2 emissions via a time-varying
carbon-intensity trace, joined by timestamp.

================================ USAGE =================================
The CLI takes a carbon-intensity trace (--carbon_trace, a parquet with
['timestamp', 'carbon_intensity'] columns; intensity in gCO2/kWh, rows on a
fixed step e.g. 30 min) and a power timeline, then bills each unit of energy.
Window-spanning intervals are split at boundaries and weighted by time (see
kavier_co2.emissions).

CONSERVATIVE DOWN-ESTIMATION: each split piece is billed at the LOWER of its own
window's intensity and the NEXT window's intensity, i.e. a moment between two
trace points takes min(left intensity, right intensity). The last trace window
(no successor) uses its own value. This deliberately under-estimates carbon.
It deviates from OpenDC, whose CarbonModel holds the earlier point's intensity
until the next point (a left-step, no min); Kavier's total is always <= that
left-step total.

Mode 1 — from a Kavier training simulation (constant-power fragment):

  kavier-co2 --from-training \
      --carbon_trace /path/to/ct1-2025-ie-carbon-intensity.parquet \
      --model_name mistral-7b-v0.1 --method lora \
      --gpu_model NVIDIA-A100-SXM4-80GB \
      --tokens_per_sample 1024 --batch_size 4 \
      --number_gpus 8 --number_nodes 1 --total_tokens 100000000 \
      --start_time "2025-06-01 00:00"

Mode 2 — from an OpenDC powerSource.parquet (must carry a 'timestamp' column
and 'energy_usage' in watt-seconds):

  kavier-co2 --powersource /path/to/powerSource.parquet \
      --carbon_trace /path/to/ct1-2025-ie-carbon-intensity.parquet

Output: total energy (kWh), total CO2 (g and kg), energy-weighted average
intensity. Add --output_csv breakdown.csv for the per-window breakdown.

Errors: a fragment whose time falls outside the trace coverage aborts with a
message naming the covered range. Timestamps must be timezone-naive on both
sides.
=======================================================================
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

import pandas as pd

from kavier_co2.emissions import EmissionResult, Fragment, compute_emissions, load_carbon_trace
from kavier_co2.fragments import fragments_from_powersource, fragments_from_training
from library.lookup import UnknownSpecError

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
    p.add_argument("--total_tokens", type=int, default=None)
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
        "total_tokens",
        "start_time",
    )
    missing = [f"--{a}" for a in required if getattr(args, a) is None]
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
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
