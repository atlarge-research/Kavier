"""Invariants of the per-node peak/idle helper kavier.sdk.cluster.core.metrics.node_activity."""

from __future__ import annotations

from kavier.sdk.cluster.core.metrics import node_activity


def test_empty_node_is_fully_idle() -> None:
    # No jobs over a 100 s window -> peak 0, idle = whole window.
    assert node_activity([], 0.0, 100.0) == (0, 100.0)


def test_single_interval_peak_and_idle() -> None:
    # One 4-GPU job busy [10,30] in a [0,100] window: peak 4, idle = 100 - 20 = 80.
    assert node_activity([(10.0, 30.0, 4)], 0.0, 100.0) == (4, 80.0)


def test_overlapping_intervals_add_at_the_peak() -> None:
    # 4 GPUs [0,20] and 3 GPUs [10,30]: overlap [10,20] -> peak 7; busy wall = [0,30] = 30; idle 70.
    peak, idle = node_activity([(0.0, 20.0, 4), (10.0, 30.0, 3)], 0.0, 100.0)
    assert peak == 7
    assert idle == 70.0


def test_adjacent_intervals_have_no_gap() -> None:
    # [0,10] then [10,20], both 2 GPUs: busy wall = 20 continuous, idle over [0,20] = 0.
    peak, idle = node_activity([(0.0, 10.0, 2), (10.0, 20.0, 2)], 0.0, 20.0)
    assert peak == 2
    assert idle == 0.0


def test_gap_between_intervals_is_idle() -> None:
    # [0,10] and [30,40], 2 GPUs each, window [0,40]: busy 20, idle 20.
    peak, idle = node_activity([(0.0, 10.0, 2), (30.0, 40.0, 2)], 0.0, 40.0)
    assert peak == 2
    assert idle == 20.0
