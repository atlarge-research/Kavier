from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cli.efficiency_cli import add_efficiency_args
from simulator.efficiency.metrics import efficiency_summary


def main() -> None:
    args = add_efficiency_args(argparse.ArgumentParser()).parse_args()

    kavier_performance_path = Path(args.kavier).expanduser()
    opendc_output_path = Path(args.opendc).expanduser()

    if not kavier_performance_path.exists():
        raise FileNotFoundError(kavier_performance_path)
    if not opendc_output_path.exists():
        raise FileNotFoundError(opendc_output_path)

    kavier_performance_df = pd.read_parquet(kavier_performance_path)
    opendc_output_df = pd.read_parquet(opendc_output_path)

    if "total_tokens" in kavier_performance_df.columns:
        total_tokens = kavier_performance_df["total_tokens"].sum()
    else:
        raise ValueError(
            "'total_tokens' column missing in tasks.parquet.\n"
            "Make sure you kept that column when writing the file"
        )

    summary = efficiency_summary(
        kavier_performance_df, opendc_output_df, total_tokens, gpu_hour_price=args.price
    )

    print("\n----------  Efficiency summary  ----------")
    for k, v in summary.items():
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
