"""Invariants of the tight-pack node-assignment primitive kavier.sdk.cluster.core.engine.place."""

from __future__ import annotations

import pytest

from kavier.sdk.cluster.core.engine import place


def test_single_node_exact_fit() -> None:
    # 8 GPUs on nodes of 8: lands entirely on one node (node 0 wins the tie).
    assert place([8, 8], 8) == [(0, 8)]


def test_spills_to_a_second_node_tight_pack() -> None:
    # 10 GPUs, 8-GPU nodes: fill node 0 fully, spill the remaining 2 -> 8+2 (partial node allowed).
    assert place([8, 8], 10) == [(0, 8), (1, 2)]


def test_least_free_node_first_best_fit() -> None:
    # Best-fit: node 0 (only 3 free) is filled first to keep node 1 roomy; output sorted by node id.
    assert place([3, 8], 8) == [(0, 3), (1, 5)]


def test_returns_none_when_it_does_not_fit() -> None:
    # sum(free)=6 < 8 -> the job does not fit.
    assert place([3, 3], 8) is None


def test_zero_request_is_empty_assignment() -> None:
    assert place([8, 8], 0) == []


def test_does_not_mutate_free() -> None:
    free = [8, 8]
    place(free, 10)
    assert free == [8, 8]


@pytest.mark.parametrize("free,gpus", [([8, 8, 8], 20), ([1, 7, 4], 9), ([5], 5)])
def test_assignment_sums_to_request_and_respects_capacity(free: list[int], gpus: int) -> None:
    assignment = place(free, gpus)
    assert assignment is not None
    assert sum(count for _, count in assignment) == gpus
    for node_id, count in assignment:
        assert 0 < count <= free[node_id]
    # sorted by node id, no duplicate nodes
    ids = [node_id for node_id, _ in assignment]
    assert ids == sorted(set(ids))
