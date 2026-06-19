"""``kavier-perf`` command-line interface: argument parsing and the validated ``PerfArgs`` model."""

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class PerfArgs(BaseModel):
    """Validated CLI arguments for a performance run: LLM/GPU names, trace path, output folder, and
    KV/prefix-cache settings."""

    llm: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    gpu: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    trace: Path
    output_folder: Path = Path("data/output_traces")

    kv_cache: str
    export_rate: float
    flush_size: int
    prefix_cache_min_tokens: int
    max_cached_prompts: int
    cache_scope: str
    prefix_cache_policy: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="Llama-3-8B", help="Name of the LLM (must match a key in LLM_SPEC_LIBRARY)")
    parser.add_argument("--gpu", default="A10", help="Name of the GPU (must match a key in GPU_SPEC_LIBRARY)")
    parser.add_argument("--trace", default="src/data/input/input_example.csv", help="Path to input trace CSV")
    parser.add_argument("--output_folder", default="src/data/output", help="Path to output folder")

    parser.add_argument("--kv_cache", choices=["on", "off"], default="on", help="Toggle KV caching (vLLM-style)")
    parser.add_argument(
        "--export_rate",
        type=float,
        default=0.1,
        help="Interval in seconds for snapshotting the simulation state (default: 0.1 seconds)",
    )
    parser.add_argument(
        "--flush_size",
        type=int,
        default=1000,
        help="Write intermediate Parquet files every N tasks (0 → keep legacy single-shot export).",
    )
    parser.add_argument(
        "--prefix_cache_min_tokens",
        type=int,
        default=1024,
        help="Prefix length (tokens) required to enter cache.",
    )
    parser.add_argument("--max_cached_prompts", type=int, default=10, help="Maximum prefixes kept (LRU).")
    parser.add_argument(
        "--cache_scope",
        choices=["session", "global"],
        default="session",
        help="Cache key includes session_id or not.",
    )
    parser.add_argument(
        "--prefix_cache_policy",
        choices=["none", "prefill", "full"],
        default="prefill",
        help="'prefill' skips prefill only, 'full' also skips decode.",
    )
    return parser


def parse_args() -> PerfArgs:
    """Parse ``sys.argv`` and return a validated ``PerfArgs``; prints validation errors and exits 1 on failure."""
    raw = _build_parser().parse_args()
    try:
        return PerfArgs.model_validate(vars(raw))
    except ValidationError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
