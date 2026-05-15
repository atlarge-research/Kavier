def add_efficiency_args(parser):
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
        default=10.0,
        help="GPU hour price in EUR (default: 10.0)",
    )
    parser.add_argument(
        "--out",
        help="Optional path to save JSON summary",
    )
    return parser
