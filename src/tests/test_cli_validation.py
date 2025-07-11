from pathlib import Path

import pytest

from performance.args import PerfArgs


def test_valid_args(tmp_path):
    ok = PerfArgs(
        llm="Llama-3-8B",
        gpu="A10",
        trace=tmp_path / "trace.csv",
        kv_cache="on",
        export_rate=0.1,
        flush_size=1,
        prefix_cache_min_tokens=1024,
        max_cached_prompts=10,
        cache_scope="session",
        prefix_cache_policy="prefill",
    )
    assert ok.llm == "Llama-3-8B"


def test_bad_gpu_name():
    with pytest.raises(ValueError):
        PerfArgs(
            llm="Llama",
            gpu="BAD NAME!",
            trace=Path("/tmp/x"),
            kv_cache="on",
            export_rate=0.1,
            flush_size=1,
            prefix_cache_min_tokens=1,
            max_cached_prompts=1,
            cache_scope="session",
            prefix_cache_policy="none",
        )
