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


def _write_jobs(path: Path) -> None:
    path.write_text("submit_s,gpus,duration_s\n0,4,10\n0,4,10\n")


def test_cluster_cli_prints_cluster_summary(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.csv"
    _write_jobs(jobs)
    proc = _run(["--jobs", str(jobs), "--policy", "fcfs", "--num-gpus", "4"])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert payload["n_jobs"] == 2
    assert payload["makespan_s"] == pytest.approx(20.0)
    assert payload["avg_wait_s"] == pytest.approx(5.0)


def test_cluster_cli_writes_per_job_csv(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.csv"
    _write_jobs(jobs)
    out = tmp_path / "per_job.csv"
    proc = _run(["--jobs", str(jobs), "--num-gpus", "4", "--out", str(out)])
    assert proc.returncode == 0, proc.stderr
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3  # header + one row per scheduled job


def test_cluster_cli_missing_jobs_file_errors(tmp_path: Path) -> None:
    proc = _run(["--jobs", str(tmp_path / "nope.csv"), "--num-gpus", "4"])
    assert proc.returncode != 0
    assert "nope.csv" in (proc.stderr + proc.stdout)
