import argparse


def add_efficiency_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--kavier",
        required=True,
        help="Path to tasks.parquet produced by the performance run (kavier-perf).",
    )
    parser.add_argument(
        "--opendc",
        required=True,
        help="Path to powerSource.parquet produced by OpenDC for the same experiment.",
    )
    parser.add_argument(
        "--price",
        type=float,
        default=10.0,
        help="GPU-hour price expressed in your currency (default: 10).",
    )
    parser.add_argument(
        "--out",
        metavar="FILE.json",
        help="If set, dump the efficiency summary as JSON beside printing it.",
    )
    return parser
