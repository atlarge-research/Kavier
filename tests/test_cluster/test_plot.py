"""``kavier.sdk.cluster.plot_timeline`` renders the cluster timeline figure.

The tests check that a non-empty file is written and that the returned stats match the schedule.
Pixels are not compared. Requires the ``[plot]`` extra (matplotlib); skipped without it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from kavier.sdk.cluster import schedule

# Two 4-GPU jobs on a 4-GPU pool serialize [0,3600] then [3600,7200]: makespan 2 h, peak GPUs 4
# (one job at a time fills the pool), peak queue 1 (the second waits).
_JOBS = [
    {"submit_s": 0, "gpus": 4, "duration_s": 3600},
    {"submit_s": 0, "gpus": 4, "duration_s": 3600},
]


def test_plot_timeline_writes_a_nonempty_pdf_and_returns_stats(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from kavier.sdk.cluster import plot_timeline

    result = schedule(_JOBS, policy="distributed-fcfs", num_nodes=1, node_gpus=4)
    out = tmp_path / "timeline.pdf"
    stats = plot_timeline(result, str(out))

    assert out.exists() and out.stat().st_size > 0  # figure written to disk
    assert stats == {"jobs": 2, "cluster_gpus": 4, "makespan_h": 2.0, "peak_gpus": 4, "peak_queue": 1}


def test_plot_timeline_writes_a_nonempty_png(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from kavier.sdk.cluster import plot_timeline

    result = schedule(_JOBS, policy="distributed-fcfs", num_nodes=1, node_gpus=4)
    out = tmp_path / "timeline.png"
    plot_timeline(result, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_timeline_keeps_the_callers_matplotlib_backend(tmp_path) -> None:
    # Runs in a fresh interpreter so the backend choice does not leak into other tests.
    pytest.importorskip("matplotlib")
    out = tmp_path / "timeline.pdf"
    code = (
        "import matplotlib\n"
        "matplotlib.use('svg')\n"
        "from kavier.sdk.cluster import plot_timeline, schedule\n"
        "jobs = [{'submit_s': 0, 'gpus': 4, 'duration_s': 3600}]\n"
        "result = schedule(jobs, policy='distributed-fcfs', num_nodes=1, node_gpus=4)\n"
        f"plot_timeline(result, {str(out)!r})\n"
        "print(matplotlib.get_backend())\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "svg"
    assert out.stat().st_size > 0
