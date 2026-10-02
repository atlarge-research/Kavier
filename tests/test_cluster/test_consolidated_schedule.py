"""End-to-end tests of the consolidated scheduling policies through ``kavier.sdk.cluster.schedule``.

``consolidated-fcfs`` and ``consolidated-backfill`` honour each job's ``nodes`` request (gang placement).
``distributed-fcfs`` and ``distributed-backfill`` ignore it and tight-pack. The tests cover the fix for
scattered single-node jobs, the oversized handling, and the spread behaviour for contrast.
"""

from __future__ import annotations

import pytest

from kavier.sdk.cluster import schedule

# A (2 GPUs, long) lands on node 0, leaving free = [6, 8]. Only node 1 has 8 free for the 8-GPU, 1-node
# job B: spread tight-packs B as 6+2, consolidated keeps it whole on node 1.
_FRAGMENTING = [
    {"job_id": "A", "submit_s": 0, "gpus": 2, "duration_s": 1000, "nodes": 1},
    {"job_id": "B", "submit_s": 0, "gpus": 8, "duration_s": 300, "nodes": 1},
]


@pytest.mark.parametrize("policy", ["consolidated-fcfs", "consolidated-backfill"])
def test_consolidated_keeps_single_node_job_whole_despite_fragmentation(policy: str) -> None:
    # With free=[6, 8], B goes whole onto node 1.
    by_id = {j.job_id: j for j in schedule(_FRAGMENTING, policy=policy, num_nodes=2, node_gpus=8).jobs}
    assert by_id["B"].nodes == ((1, 8),)


def test_spread_backfill_still_splits_the_same_job() -> None:
    # Spread backfill tight-packs B least-free-first as 6+2.
    by_id = {j.job_id: j for j in schedule(_FRAGMENTING, policy="distributed-backfill", num_nodes=2, node_gpus=8).jobs}
    assert by_id["B"].nodes == ((0, 6), (1, 2))


@pytest.mark.parametrize(
    "gpus,nodes,per_node",
    [
        (8, 1, 8),  # one whole node
        (12, 2, 6),  # even split, 6 per node
        (6, 3, 2),  # thin replicas on three nodes
    ],
)
def test_feasible_job_uses_exactly_n_distinct_nodes_evenly(gpus: int, nodes: int, per_node: int) -> None:
    # On an empty 8x8 cluster a feasible job occupies `nodes` distinct nodes with gpus // nodes GPUs each.
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
    # gpus=24, nodes=3 on an empty 4x8 cluster: 8/8/8 on nodes 0-2; node 3 stays idle.
    res = schedule(
        [{"job_id": "j", "submit_s": 0, "gpus": 24, "duration_s": 600, "nodes": 3}],
        policy="consolidated-backfill",
        num_nodes=4,
        node_gpus=8,
    )
    assert res.jobs[0].nodes == ((0, 8), (1, 8), (2, 8))


@pytest.mark.parametrize("policy", ["consolidated-fcfs", "consolidated-backfill"])
def test_consolidated_drop_skips_only_jobs_larger_than_the_cluster(policy: str) -> None:
    # On a 4x8 cluster, 40 GPUs exceed the 32-GPU capacity and are dropped. 16 GPUs on 1 node cannot
    # fit one 8-GPU node but fit the cluster, so drop widens that job to 2 nodes as cap does.
    jobs = [
        {"job_id": "toobig", "submit_s": 0, "gpus": 40, "duration_s": 300, "nodes": 1},
        {"job_id": "wide", "submit_s": 0, "gpus": 16, "duration_s": 300, "nodes": 1},
        {"job_id": "ok", "submit_s": 0, "gpus": 8, "duration_s": 300, "nodes": 1},
    ]
    res = schedule(jobs, policy=policy, num_nodes=4, node_gpus=8, oversized="drop")
    assert res.dropped == ["toobig"]
    by_id = {j.job_id: j for j in res.jobs}
    assert by_id["wide"].gpus == 16
    assert by_id["wide"].nodes == ((0, 8), (1, 8))
    assert by_id["ok"].nodes == ((2, 8),)


def test_default_policy_with_drop_runs_a_multi_node_job_without_a_nodes_column() -> None:
    # Without a nodes column every job asks for 1 node. A 16-GPU job still fits a 4x8 cluster.
    res = schedule([{"submit_s": 0, "gpus": 16, "duration_s": 10}], num_nodes=4, node_gpus=8, oversized="drop")
    assert res.dropped == []
    assert res.jobs[0].nodes == ((0, 8), (1, 8))


def test_consolidated_cap_widens_nodes_so_the_share_fits() -> None:
    # A 16-GPU, 1-node request cannot fit one 8-GPU node. oversized="cap" widens it to
    # min_nodes = ceil(16/8) = 2 and runs it as 8+8 with gpus kept at 16.
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
    # gpus=999 on a 4x8 cluster: cap clamps to the 32-GPU capacity, 8 on each of the 4 nodes.
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
