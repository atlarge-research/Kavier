"""Tests for ``kavier.sdk.inference.core.service.run_performance`` called as a Python API.

Checks the files written to the timestamped output folder, their columns against the OpenDC schemas,
and row counts against the input trace. Prefill and decode timings are covered in test_runner.py.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest

from kavier.cli.inference import PerfArgs
from kavier.sdk.inference.core import service
from kavier.sdk.inference.core.service import run_performance
from kavier.sdk.io.opendc.schema import FRAGMENTS_SCHEMA, TASKS_SCHEMA

EXAMPLE_TRACE = Path(str(files("kavier.sdk.inference").joinpath("data", "input", "input_example.csv")))


def _args(output_folder: Path) -> PerfArgs:
    # Defaults of kavier/cli/inference.py::_build_parser, except llm, gpu, trace and output_folder.
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
    """Run ``run_performance`` once on the shipped example trace and share the output."""
    out_root = tmp_path_factory.mktemp("service_run")
    results_text = run_performance(_args(out_root))
    run_dirs = [p for p in out_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1  # one timestamped folder per call
    return _Run(out_dir=run_dirs[0], results_text=results_text)


@pytest.fixture(scope="module")
def trace_oracle() -> tuple[int, int]:
    """Return (row_count, total_tokens) read from the CSV, without the simulation engine."""
    csv = pd.read_csv(EXAMPLE_TRACE)
    n_rows = len(csv)
    total_tokens = int((csv["num_input_tokens"] + csv["num_output_tokens"]).sum())
    return n_rows, total_tokens


def test_shipped_trace_fixture_has_the_expected_shape(trace_oracle: tuple[int, int]) -> None:
    # Guards against an edit of input_example.csv. 84 = sum of per-row input+output tokens:
    # (8+6) + (12+5) + (6+9) + (10+4) + (5+7) + (9+3).
    assert trace_oracle == (6, 84)


def test_run_performance_writes_the_three_expected_files(run: _Run) -> None:
    names = {p.name for p in run.out_dir.iterdir()}
    assert names == {"tasks.parquet", "fragments.parquet", "_sim_results.txt"}


def test_run_performance_return_value_is_the_results_text_and_matches_sidecar(run: _Run) -> None:
    # service.py writes the returned string to _sim_results.txt unchanged.
    # The banner line comes from core/metrics.py::Metrics.summary.
    assert (run.out_dir / "_sim_results.txt").read_text() == run.results_text
    assert isinstance(run.results_text, str)
    assert "SIMULATION SUMMARY" in run.results_text


def test_tasks_parquet_matches_opendc_schema_and_trace_row_count(run: _Run, trace_oracle: tuple[int, int]) -> None:
    n_rows, total_tokens = trace_oracle
    table = pq.read_table(run.out_dir / "tasks.parquet")
    # OpenDC tasks schema plus total_tokens, which adapter.py appends for inference runs.
    assert table.schema.names == list(TASKS_SCHEMA.names) + ["total_tokens"]

    tasks = table.to_pandas()
    # One task per trace row.
    assert len(tasks) == n_rows
    # No tokens dropped or added between the CSV and the parquet.
    assert int(tasks["total_tokens"].sum()) == total_tokens
    # duration [ms] is floored at 1 by runner.py: max(1, int(round(total_s * 1000))).
    assert (tasks["duration"] >= 1).all()


def test_fragments_parquet_matches_opendc_schema_and_tiles_the_tasks(run: _Run, trace_oracle: tuple[int, int]) -> None:
    n_rows, _ = trace_oracle
    table = pq.read_table(run.out_dir / "fragments.parquet")
    assert table.schema.names == list(FRAGMENTS_SCHEMA.names)

    frags = table.to_pandas()
    # runner.py emits at least one fragment per task.
    assert len(frags) >= n_rows
    # Every fragment id is a task id in 0..n_rows-1.
    assert set(frags["id"].unique()) <= set(range(n_rows))
    # adapter.py converts duration to a Timedelta; runner.py floors each fragment at 1 ms.
    assert (frags["duration"] > pd.Timedelta(0)).all()


class _FrozenDatetime(datetime.datetime):
    """``datetime`` whose ``now()`` always returns the same second."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 3, 4, 5, 6, 7)


def test_runs_started_in_the_same_second_get_separate_folders(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(service, "datetime", SimpleNamespace(datetime=_FrozenDatetime))
    run_performance(_args(tmp_path))
    run_performance(_args(tmp_path))
    names = sorted(p.name for p in tmp_path.iterdir())
    # The first run keeps the plain timestamp name; the second gets a suffix.
    assert names == ["2026-03-04_05-06-07", "2026-03-04_05-06-07_1"]
    for name in names:
        assert {p.name for p in (tmp_path / name).iterdir()} == {
            "tasks.parquet",
            "fragments.parquet",
            "_sim_results.txt",
        }


def test_identical_runs_write_identical_opendc_workloads(tmp_path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    run_performance(_args(first))
    run_performance(_args(second))
    (dir_a,) = first.iterdir()
    (dir_b,) = second.iterdir()
    for name in ("tasks.parquet", "fragments.parquet"):
        assert pq.read_table(dir_a / name).equals(pq.read_table(dir_b / name))
    # Tasks are stamped from a fixed origin, 2026-01-01 00:00 UTC.
    submitted = pq.read_table(dir_a / "tasks.parquet").column("submission_time").to_pylist()
    assert set(submitted) == {datetime.datetime(2026, 1, 1)}
