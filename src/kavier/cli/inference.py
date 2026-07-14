"""``kavier inference`` subcommand: parse and validate CLI args, then run the per-request simulator."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from kavier.sdk.defaults import (
    DEFAULT_CACHE_SCOPE,
    DEFAULT_CLI_PREFIX_POLICY,
    DEFAULT_EXPORT_RATE,
    DEFAULT_INFERENCE_GPU,
    DEFAULT_INFERENCE_MODEL,
    DEFAULT_PREFIX_MIN_TOKENS,
)
from kavier.sdk.inference.core.config import CacheAction, CacheScope
from kavier.sdk.inference.core.service import run_performance
from kavier.sdk.library.lookup import UnknownSpecError

#: Tiny example trace shipped inside the ``kavier.sdk.inference`` package (the uv_build wheel ships it);
#: resolved via importlib.resources so a bare ``kavier inference`` works from any working directory.
DEFAULT_TRACE = Path(str(files("kavier.sdk.inference").joinpath("data", "input", "input_example.csv")))

#: Default output folder, created relative to the current working directory.
DEFAULT_OUTPUT_FOLDER = Path("kavier_output")

#: Spec-library key: words of [A-Za-z0-9._-] separated by single spaces (e.g. "H200 SXM"),
#: with no leading/trailing whitespace.
_SPEC_KEY_PATTERN = r"^[A-Za-z0-9._-]+( [A-Za-z0-9._-]+)*$"


class PerfArgs(BaseModel):
    """Validated CLI arguments for ``kavier inference``."""

    llm: str = Field(pattern=_SPEC_KEY_PATTERN)
    gpu: str = Field(pattern=_SPEC_KEY_PATTERN)
    trace: Path
    output_folder: Path = DEFAULT_OUTPUT_FOLDER

    kv_cache: str
    export_rate: float = Field(gt=0)  # snapshot interval; runner divides by it (0 → ZeroDivisionError)
    flush_size: int = Field(ge=0)  # 0 keeps the single-shot export
    prefix_cache_min_tokens: int = Field(ge=0)
    max_cached_prompts: int = Field(ge=1)  # LRUCache(maxsize=0) raises KeyError on first insert
    cache_scope: str
    prefix_cache_policy: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kavier inference")
    parser.add_argument(
        "--llm", default=DEFAULT_INFERENCE_MODEL, help="Name of the LLM (must match a key in LLM_SPEC_LIBRARY)"
    )
    parser.add_argument(
        "--gpu", default=DEFAULT_INFERENCE_GPU, help="Name of the GPU (must match a key in GPU_SPEC_LIBRARY)"
    )
    parser.add_argument(
        "--trace",
        default=DEFAULT_TRACE,
        help="Path to input trace CSV (default: the packaged example trace, %(default)s)",
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        help="Path to output folder, created if missing (default: %(default)s in the current working directory)",
    )

    parser.add_argument("--kv_cache", choices=["on", "off"], default="on", help="Toggle KV caching (vLLM-style)")
    parser.add_argument(
        "--export_rate",
        type=float,
        default=DEFAULT_EXPORT_RATE,
        help="Interval in seconds for snapshotting the simulation state (default: 0.1 seconds)",
    )
    parser.add_argument(
        "--flush_size",
        type=int,
        default=1000,
        help="Write intermediate Parquet files every N tasks (0 → keep the single-shot export).",
    )
    parser.add_argument(
        "--prefix_cache_min_tokens",
        type=int,
        default=DEFAULT_PREFIX_MIN_TOKENS,
        help="Prefix length (tokens) required to enter cache.",
    )
    parser.add_argument("--max_cached_prompts", type=int, default=10, help="Maximum prefixes kept (LRU).")
    parser.add_argument(
        "--cache_scope",
        choices=[s.value for s in CacheScope],
        default=DEFAULT_CACHE_SCOPE,
        help="Cache key includes session_id or not.",
    )
    parser.add_argument(
        "--prefix_cache_policy",
        choices=[a.value for a in CacheAction],
        default=DEFAULT_CLI_PREFIX_POLICY,
        help="'prefill' skips prefill only, 'full' also skips decode.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> PerfArgs:
    """Parse and validate ``kavier inference`` CLI arguments; exits with error on validation failure."""
    raw = _build_parser().parse_args(argv)
    try:
        return PerfArgs.model_validate(vars(raw))
    except ValidationError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the inference simulator: parse args, then simulate and export (OpenDC + spec outputs)."""
    args = parse_args(argv)
    try:
        run_performance(args)
    except UnknownSpecError as exc:
        print(f"kavier inference: error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
