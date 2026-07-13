"""End-to-end contract for ``kavier cluster`` (shells out to the real CLI).

Oracles are hand-derived from the schedule, never copied from a run: two 4-GPU jobs on a 4-GPU pool
serialize to [0,10] and [10,20] s, so makespan = 20 s and average wait = (0+10)/2 = 5 s.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "kavier.cli", "cluster", *args], capture_output=True, text=True)


def _write_jobs(tmp_path: Path) -> Path:
    jobs = tmp_path / "jobs.csv"
    jobs.write_text("submit_s,gpus,duration_s\n0,4,10\n0,4,10\n")
    return jobs


def test_cluster_cli_prints_cluster_summary(tmp_path: Path) -> None:
    jobs = _write_jobs(tmp_path)
    proc = _run(["--jobs", str(jobs), "--policy", "fcfs", "--num-nodes", "1", "--node-gpus", "4"])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert payload["n_jobs"] == 2
    assert payload["makespan_s"] == pytest.approx(20.0)
    assert payload["avg_wait_s"] == pytest.approx(5.0)


def test_cluster_cli_writes_per_job_csv(tmp_path: Path) -> None:
    jobs = _write_jobs(tmp_path)
    out = tmp_path / "per_job.csv"
    proc = _run(["--jobs", str(jobs), "--num-nodes", "1", "--node-gpus", "4", "--out", str(out)])
    assert proc.returncode == 0, proc.stderr
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3  # header + one row per scheduled job


def test_cluster_cli_missing_jobs_file_errors(tmp_path: Path) -> None:
    proc = _run(["--jobs", str(tmp_path / "nope.csv"), "--num-nodes", "1", "--node-gpus", "4"])
    assert proc.returncode != 0
    assert "nope.csv" in (proc.stderr + proc.stdout)


def test_cluster_cli_renders_timeline_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    jobs = _write_jobs(tmp_path)
    out = tmp_path / "timeline.pdf"
    proc = _run(["--jobs", str(jobs), "--num-nodes", "1", "--node-gpus", "4", "--plot", str(out)])
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 0  # the figure was rendered by the CLI


def test_backfill_with_node_topology_runs(tmp_path: Path) -> None:
    jobs = _write_jobs(tmp_path)  # reuse the file-writing helper already in this module
    proc = _run(["--jobs", str(jobs), "--policy", "backfill", "--num-nodes", "2", "--node-gpus", "8"])
    assert proc.returncode == 0
    summary = json.loads(proc.stdout)
    assert summary["policy"] == "backfill"
    assert summary["capacity_gpus"] == 16


def test_per_job_csv_has_nodes_column(tmp_path: Path) -> None:
    jobs = _write_jobs(tmp_path)
    out = tmp_path / "per_jobs.csv"
    proc = _run(["--jobs", str(jobs), "--num-nodes", "2", "--node-gpus", "8", "--out", str(out)])
    assert proc.returncode == 0
    header = out.read_text().splitlines()[0]
    assert "nodes" in header.split(",")


def test_per_node_csv_is_written(tmp_path: Path) -> None:
    jobs = _write_jobs(tmp_path)
    out_nodes = tmp_path / "per_nodes.csv"
    proc = _run(["--jobs", str(jobs), "--num-nodes", "2", "--node-gpus", "8", "--out-nodes", str(out_nodes)])
    assert proc.returncode == 0
    lines = out_nodes.read_text().splitlines()
    assert lines[0].split(",") == [
        "node_id", "gpus", "jobs_hosted", "busy_gpu_s", "utilization", "peak_gpus_used", "idle_s", "energy_kwh"
    ]
    assert len(lines) == 1 + 2  # header + 2 node rows


def test_missing_topology_errors_friendly(tmp_path: Path) -> None:
    jobs = _write_jobs(tmp_path)
    proc = _run(["--jobs", str(jobs)])  # no --num-nodes/--node-gpus
    assert proc.returncode == 2
    assert "num_nodes" in proc.stderr
