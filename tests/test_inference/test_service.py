"""Wiring pin for ``kavier.sdk.inference.core.service.run_performance`` (zero-coverage module,
de-slop Phase 0 Deliverable B).

``run_performance`` has no direct test today: it is only exercised indirectly through the ``kavier
inference`` CLI subprocess (test_integration/test_cli_contract.py). This module calls it directly as a
Python API -- args in, timestamped OpenDC-shaped output folder out -- and pins the wiring: which files
get written, their column names (from the OpenDC schemas, not re-typed by hand), and row counts against
the input trace. It intentionally does NOT re-derive prefill/decode timings -- that physics is owned by
test_inference/test_runner.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from kavier.cli.inference import PerfArgs
from kavier.sdk.inference.core.service import run_performance
from kavier.sdk.io.opendc.schema import FRAGMENTS_SCHEMA, TASKS_SCHEMA

EXAMPLE_TRACE = Path(str(files("kavier.sdk.inference").joinpath("data", "input", "input_example.csv")))


def _args(output_folder: Path) -> PerfArgs:
    # Mirrors kavier/cli/inference.py::_build_parser's defaults exactly (the values run_performance
    # is actually invoked with in production), except llm/gpu/trace/output_folder which we pin here.
    return PerfArgs(
        llm="Llama-3-8B",
        gpu="A10",
        trace=EXAMPLE_TRACE,
        output_folder=output_folder,
        kv_cache="on",
        export_rate=0.1,
        flush_size=1000,
        prefix_cache_min_tokens=1024,
        max_cached_prompts=10,
        cache_scope="session",
        prefix_cache_policy="prefill",
    )


@dataclass(frozen=True)
class _Run:
    out_dir: Path
    results_text: str


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> _Run:
    """Run ``run_performance`` once against the shipped example trace; every test reads its output.

    One call only (not one per test): service.py names the run folder with SECOND resolution
    (``%Y-%m-%d_%H-%M-%S``), so two calls issued back-to-back in the same wall-clock second would
    collide on the same directory -- a real trap, sidestepped here rather than pinned as a "feature".
    """
    out_root = tmp_path_factory.mktemp("service_run")
    results_text = run_performance(_args(out_root))
    run_dirs = [p for p in out_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1  # one timestamped folder per invocation (service.py's own contract)
    return _Run(out_dir=run_dirs[0], results_text=results_text)


@pytest.fixture(scope="module")
def trace_oracle() -> tuple[int, int]:
    """(row_count, total_tokens) read straight from the CSV -- independent of the simulation engine."""
    csv = pd.read_csv(EXAMPLE_TRACE)
    n_rows = len(csv)
    total_tokens = int((csv["num_input_tokens"] + csv["num_output_tokens"]).sum())
    return n_rows, total_tokens


def test_shipped_trace_fixture_has_the_expected_shape(trace_oracle: tuple[int, int]) -> None:
    # Guards the two fixtures below against a silent edit of input_example.csv (same guard style as
    # test_integration/test_cli_contract.py's inference test). 84 = per-row input+output token sums
    # (8+6) + (12+5) + (6+9) + (10+4) + (5+7) + (9+3) over the 6 rows.
    assert trace_oracle == (6, 84)


def test_run_performance_writes_the_three_expected_files(run: _Run) -> None:
    names = {p.name for p in run.out_dir.iterdir()}
    assert names == {"tasks.parquet", "fragments.parquet", "_sim_results.txt"}


def test_run_performance_return_value_is_the_results_text_and_matches_sidecar(run: _Run) -> None:
    # service.py: results = simulate(...); ...; output_kavier_specs(out_dir, results); return results.
    # The sidecar file and the return value must be the exact same string (one write, no copy/format
    # drift), and core/metrics.py::Metrics.summary always emits this fixed banner line.
    assert (run.out_dir / "_sim_results.txt").read_text() == run.results_text
    assert isinstance(run.results_text, str)
    assert "SIMULATION SUMMARY" in run.results_text


def test_tasks_parquet_matches_opendc_schema_and_trace_row_count(run: _Run, trace_oracle: tuple[int, int]) -> None:
    n_rows, total_tokens = trace_oracle
    table = pq.read_table(run.out_dir / "tasks.parquet")
    # Column set: the OpenDC tasks schema plus total_tokens (inference-only field; adapter.py appends
    # it only when present -- confirmed by test_integration/test_opendc_adapter.py, not re-derived here).
    assert table.schema.names == list(TASKS_SCHEMA.names) + ["total_tokens"]

    tasks = table.to_pandas()
    # One task per trace row (runner.py::simulate_one is called once per request).
    assert len(tasks) == n_rows
    # Per-task total_tokens sums to the same total the raw CSV gives (num_input+num_output), so the
    # engine neither drops nor invents tokens on the way to the parquet.
    assert int(tasks["total_tokens"].sum()) == total_tokens
    # duration (ms) is floored at 1 by runner.py's `max(1, int(round(total_s * 1000)))` -- every row.
    assert (tasks["duration"] >= 1).all()


def test_fragments_parquet_matches_opendc_schema_and_tiles_the_tasks(run: _Run, trace_oracle: tuple[int, int]) -> None:
    n_rows, _ = trace_oracle
    table = pq.read_table(run.out_dir / "fragments.parquet")
    assert table.schema.names == list(FRAGMENTS_SCHEMA.names)

    frags = table.to_pandas()
    # runner.py emits max(1, ...) fragments per task -> at least one fragment per task overall.
    assert len(frags) >= n_rows
    # Every fragment's id is one of the task ids (0..n_rows-1): fragments belong to a real task, none
    # orphaned and none out of range.
    assert set(frags["id"].unique()) <= set(range(n_rows))
    # duration is coerced to a pandas Timedelta (adapter.py: pd.to_timedelta(..., unit="ms")); runner.py
    # floors every fragment's duration_ms at 1, so none can be zero/negative after coercion.
    assert (frags["duration"] > pd.Timedelta(0)).all()
