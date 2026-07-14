"""Invariants of the consolidated (gang) placement primitive
``kavier.sdk.cluster.core.engine.place_consolidated``.

``place_consolidated(free, gpus, nodes, node_gpus)`` must put ``gpus`` GPUs on EXACTLY
``max(1, min(nodes, len(free)))`` distinct nodes, even-split, never scattered wider — the fix for the
split-placement bug where a nodes=1 job was fragmented across several nodes.
"""

from __future__ import annotations

from kavier.sdk.cluster.core.engine import place_consolidated


def test_single_node_job_is_never_split() -> None:
    # The reported bug: gpus=8, nodes=1 must land entirely on ONE node, never scattered (e.g. 7+1).
    # n_eff=min(1,4)=1 -> demands=[8]; 8 <= node_gpus=8 fits. Tightest node with >=8 free is node 0.
    assert place_consolidated([8, 8, 8, 8], 8, 1, 8) == [(0, 8)]


def test_even_split_across_exactly_n_nodes() -> None:
    # gpus=24, nodes=3, node_gpus=8: divmod(24,3)=(8,0) -> demands [8,8,8]. Three distinct 8-GPU
    # replicas; tightest-fit fills nodes 0,1,2 on a fresh 4x8 cluster and leaves node 3 idle.
    assert place_consolidated([8, 8, 8, 8], 24, 3, 8) == [(0, 8), (1, 8), (2, 8)]


def test_infeasible_when_per_node_share_exceeds_node_gpus() -> None:
    # gpus=16, nodes=1, node_gpus=8: a single replica would need 16 GPUs on one 8-GPU node.
    # max(demands)=16 > 8 -> infeasible regardless of how much total room exists.
    assert place_consolidated([8, 8, 8, 8], 16, 1, 8) is None


def test_uneven_split_sums_to_exactly_gpus() -> None:
    # gpus=10, nodes=3: divmod(10,3)=(3,1) -> demands [4,3,3] summing to 10 exactly (never 3+3+3=9).
    assignment = place_consolidated([8, 8, 8, 8], 10, 3, 8)
    assert assignment is not None
    assert len(assignment) == 3  # exactly n_eff distinct nodes
    assert sum(count for _, count in assignment) == 10  # reserves EXACTLY the request
    assert sorted(count for _, count in assignment) == [3, 3, 4]  # the even split, one node gets the +1


def test_nodes_clamped_to_available_when_more_requested_than_exist() -> None:
    # nodes=10 requested but only 2 nodes exist: n_eff=min(10,2)=2 -> divmod(8,2)=(4,0) -> [4,4].
    # (A request for more nodes than exist is clamped, not rejected.)
    assert place_consolidated([8, 8], 8, 10, 8) == [(0, 4), (1, 4)]


def test_tightest_fit_keeps_roomy_nodes_open() -> None:
    # gpus=8, nodes=1 on free [3, 8, 10]: nodes with >=8 free are 1 (8) and 2 (10). Tightest-fit takes
    # the least-free qualifier (node 1) so the roomy node 2 stays open for a future large job.
    assert place_consolidated([3, 8, 10], 8, 1, 8) == [(1, 8)]


def test_returns_none_when_not_enough_distinct_free_nodes_now() -> None:
    # gpus=16, nodes=2, node_gpus=8: demands [8,8] need TWO distinct nodes each with >=8 free.
    # free=[8,4,4]: only node 0 qualifies, so the second replica has nowhere to go -> None (no
    # fragmenting a replica across the two 4-free nodes).
    assert place_consolidated([8, 4, 4], 16, 2, 8) is None


def test_does_not_mutate_free() -> None:
    free = [8, 8, 8, 8]
    place_consolidated(free, 24, 3, 8)
    assert free == [8, 8, 8, 8]


def test_zero_request_is_empty_assignment() -> None:
    assert place_consolidated([8, 8], 0, 1, 8) == []
