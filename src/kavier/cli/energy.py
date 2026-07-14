"""``kavier energy`` subcommand: per-Mtoken efficiency from a Kavier performance file and OpenDC power output."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from kavier.sdk.energy.metrics import efficiency_summary


def add_efficiency_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``kavier energy`` flags to ``parser``."""
    parser.add_argument("--kavier", required=True, help="Path to Kavier performance output (tasks.parquet)")
    parser.add_argument("--opendc", required=True, help="Path to OpenDC output (powerSource.parquet)")
    parser.add_argument(
        "--price",
        type=float,
        default=None,
        help="GPU-hour price for the $/token metric. No default -- financial efficiency "
        "is reported only when you set it (the GPU cost is yours to specify).",
    )
    parser.add_argument("--out", help="Optional path to save JSON summary")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Compute and print per-Mtoken efficiency from Kavier + OpenDC output."""
    args = add_efficiency_args(argparse.ArgumentParser(prog="kavier energy")).parse_args(argv)

    kavier_performance_path = Path(args.kavier).expanduser()
    opendc_output_path = Path(args.opendc).expanduser()

    if not kavier_performance_path.exists():
        raise FileNotFoundError(kavier_performance_path)
    if not opendc_output_path.exists():
        raise FileNotFoundError(opendc_output_path)

    kavier_performance_df = pd.read_parquet(kavier_performance_path)
    opendc_output_df = pd.read_parquet(opendc_output_path)

    if "total_tokens" not in kavier_performance_df.columns:
        raise ValueError(
            "'total_tokens' column missing in tasks.parquet.\nMake sure you kept that column when writing the file"
        )
    total_tokens = kavier_performance_df["total_tokens"].sum()

    summary = efficiency_summary(kavier_performance_df, opendc_output_df, total_tokens, gpu_hour_price=args.price)

    print("\n----------  Efficiency summary  ----------")
    for k, v in summary.items():
        if v is None:
            print(f"{k:>42}: N/A  (pass --price to compute)")
        else:
            print(f"{k:>42}: {v:,.6g}")
    print("------------------------------------------")

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(summary, f, indent=4)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
