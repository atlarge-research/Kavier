"""``kavier.sdk.cluster.plot_timeline`` renders the operational cluster-timeline figure.

We don't assert pixels — the falsifiable behaviour is "a non-empty figure file is written" (a broken
renderer throws or writes nothing) plus a stats dict that matches the simulated schedule. Needs the
``[plot]`` extra (matplotlib); skipped otherwise.
"""

from __future__ import annotations

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

    result = schedule(_JOBS, policy="fcfs", num_gpus=4)
    out = tmp_path / "timeline.pdf"
    stats = plot_timeline(result, str(out))

    assert out.exists() and out.stat().st_size > 0  # the figure was actually rendered to disk
    assert stats == {"jobs": 2, "cluster_gpus": 4, "makespan_h": 2.0, "peak_gpus": 4, "peak_queue": 1}


def test_plot_timeline_writes_a_nonempty_png(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from kavier.sdk.cluster import plot_timeline

    result = schedule(_JOBS, policy="fcfs", num_gpus=4)
    out = tmp_path / "timeline.png"
    plot_timeline(result, str(out))
    assert out.exists() and out.stat().st_size > 0
