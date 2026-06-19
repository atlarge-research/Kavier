"""Argparse wiring for the ``kavier-energy`` efficiency CLI."""


def add_efficiency_args(parser):
    """Register the ``kavier-energy`` arguments (--kavier, --opendc, --price, --out)
    on ``parser`` and return it."""
    parser.add_argument(
        "--kavier",
        required=True,
        help="Path to Kavier performance output (tasks.parquet)",
    )
    parser.add_argument(
        "--opendc",
        required=True,
        help="Path to OpenDC output (powerSource.parquet)",
    )
    parser.add_argument(
        "--price",
        type=float,
        default=None,
        help="GPU-hour price for the $/token metric. No default -- financial efficiency "
             "is reported only when you set it (the GPU cost is yours to specify).",
    )
    parser.add_argument(
        "--out",
        help="Optional path to save JSON summary",
    )
    return parser
