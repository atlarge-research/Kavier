import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from cli.performance_cli import add_performance_args


class PerfArgs(BaseModel):
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


def parse_args() -> PerfArgs:
    raw = add_performance_args(argparse.ArgumentParser()).parse_args()
    try:
        return PerfArgs.model_validate(vars(raw))
    except ValidationError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
