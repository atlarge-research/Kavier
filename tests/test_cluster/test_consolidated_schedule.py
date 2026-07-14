"""End-to-end behaviour of the consolidated scheduling policies via ``kavier.sdk.cluster.schedule``.

``consolidated-fcfs`` / ``consolidated-backfill`` honour each job's ``nodes`` request (gang
placement); ``distributed-fcfs`` / ``distributed-backfill`` ignore it (tight-pack). These tests pin the
reported bug fix (a
single-node job must never be scattered) and the oversized handling, and contrast against the
unchanged spread behaviour.
"""

from __future__ import annotations

import pytest

from kavier.sdk.cluster import schedule

# A (2 GPUs, long) lands on node 0 first, leaving free = [6, 8]. A later 8-GPU/1-node job B then
# has only ONE node (node 1) with a full 8 free — spread would tight-pack it as 6+2, consolidated
# must keep it whole on node 1.
_FRAGMENTING = [
    {"job_id": "A", "submit_s": 0, "gpus": 2, "duration_s": 1000, "nodes": 1},
    {"job_id": "B", "submit_s": 0, "gpus": 8, "duration_s": 300, "nodes": 1},
]


@pytest.mark.parametrize("policy", ["consolidated-fcfs", "consolidated-backfill"])
def test_consolidated_keeps_single_node_job_whole_despite_fragmentation(policy: str) -> None:
    # The reported bug, reproduced end-to-end: with node 0 partly used (free=[6,8]), B's 8-GPU/1-node
    # request must occupy the one node with a full 8 free (node 1), NOT be scattered 6+2.
    by_id = {j.job_id: j for j in schedule(_FRAGMENTING, policy=policy, num_nodes=2, node_gpus=8).jobs}
    assert by_id["B"].nodes == ((1, 8),)


def test_spread_backfill_still_splits_the_same_job() -> None:
    # Contrast + regression pin: plain backfill (spread) DOES scatter B as 6+2 (least-free-first
    # tight-pack) — the exact bug the consolidated policies fix, and proof the spread path is unchanged.
    by_id = {j.job_id: j for j in schedule(_FRAGMENTING, policy="distributed-backfill", num_nodes=2, node_gpus=8).jobs}
    assert by_id["B"].nodes == ((0, 6), (1, 2))


@pytest.mark.parametrize(
    "gpus,nodes,per_node",
    [
        (8, 1, 8),  # the bug case: one whole node
        (12, 2, 6),  # even split, not a full node each
        (6, 3, 2),  # thin replicas, three nodes
    ],
)
def test_feasible_job_uses_exactly_n_distinct_nodes_evenly(gpus: int, nodes: int, per_node: int) -> None:
    # A single feasible job on an empty 8x8 cluster occupies EXACTLY `nodes` distinct nodes, each
    # holding gpus//nodes GPUs (one replica per node, even split).
    res = schedule(
        [{"submit_s": 0, "gpus": gpus, "duration_s": 100, "nodes": nodes}],
        policy="consolidated-fcfs",
        num_nodes=8,
        node_gpus=8,
    )
    placement = res.jobs[0].nodes
    assert len(placement) == nodes
    assert all(count == per_node for _, count in placement)
    assert sum(count for _, count in placement) == gpus


def test_consolidated_places_24gpu_3node_job_on_exactly_three_nodes() -> None:
    # gpus=24, nodes=3 on an empty 4x8 cluster: even split 8/8/8 across three distinct nodes (node 3
    # stays idle), never spread across all four.
    res = schedule(
        [{"job_id": "j", "submit_s": 0, "gpus": 24, "duration_s": 600, "nodes": 3}],
        policy="consolidated-backfill",
        num_nodes=4,
        node_gpus=8,
    )
    assert res.jobs[0].nodes == ((0, 8), (1, 8), (2, 8))


def test_consolidated_drops_job_whose_per_node_share_cannot_fit() -> None:
    # gpus=16, nodes=1 on 8-GPU nodes: one replica needs 16 GPUs on a single 8-GPU node -> infeasible.
    # oversized="drop" excludes it and reports it; the co-submitted feasible job survives.
    jobs = [
        {"job_id": "toobig", "submit_s": 0, "gpus": 16, "duration_s": 300, "nodes": 1},
        {"job_id": "ok", "submit_s": 0, "gpus": 8, "duration_s": 300, "nodes": 1},
    ]
    res = schedule(jobs, policy="consolidated-backfill", num_nodes=4, node_gpus=8, oversized="drop")
    assert res.dropped == ["toobig"]
    assert [j.job_id for j in res.jobs] == ["ok"]


def test_consolidated_cap_widens_nodes_so_the_share_fits() -> None:
    # oversized="cap": a 16-GPU/1-node request can't fit one 8-GPU node. Cap widens nodes just enough
    # (min_nodes=ceil(16/8)=2) so it runs as 8+8 on two nodes — not dropped, gpus preserved at 16.
    res = schedule(
        [{"job_id": "j", "submit_s": 0, "gpus": 16, "duration_s": 300, "nodes": 1}],
        policy="consolidated-backfill",
        num_nodes=4,
        node_gpus=8,
        oversized="cap",
    )
    assert res.dropped == []
    job = res.jobs[0]
    assert job.gpus == 16
    assert job.nodes == ((0, 8), (1, 8))


def test_consolidated_cap_clamps_gpus_to_cluster_capacity() -> None:
    # gpus=999 on a 4x8 (32-GPU) cluster: cap clamps to the 32-GPU capacity and spreads it as evenly
    # as the geometry allows -> 8 on every one of the 4 nodes.
    res = schedule(
        [{"job_id": "j", "submit_s": 0, "gpus": 999, "duration_s": 300, "nodes": 1}],
        policy="consolidated-backfill",
        num_nodes=4,
        node_gpus=8,
        oversized="cap",
    )
    job = res.jobs[0]
    assert job.gpus == 32
    assert job.nodes == ((0, 8), (1, 8), (2, 8), (3, 8))
