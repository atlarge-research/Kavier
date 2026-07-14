"""Per-node output (NodeRecord): assignment, utilisation, energy conservation, idle nodes."""

from __future__ import annotations

import pytest

from kavier.sdk.cluster import schedule


def test_one_row_per_node_including_idle_nodes() -> None:
    # A single tiny job on a 4-node cluster still yields 4 node rows; 3 are fully idle.
    res = schedule([{"submit_s": 0, "gpus": 2, "duration_s": 10}], policy="distributed-fcfs", num_nodes=4, node_gpus=8)
    assert [n.node_id for n in res.nodes] == [0, 1, 2, 3]
    assert res.nodes[0].jobs_hosted == 1
    assert [n.jobs_hosted for n in res.nodes[1:]] == [0, 0, 0]
    assert res.nodes[0].peak_gpus_used == 2


def test_node_busy_seconds_sum_to_total_gpu_seconds() -> None:
    # Two 8-GPU jobs co-run on a 2x8 cluster for 10 s: each node busy 8x10 = 80 GPU·s; sum 160.
    jobs = [{"submit_s": 0, "gpus": 8, "duration_s": 10}, {"submit_s": 0, "gpus": 8, "duration_s": 10}]
    res = schedule(jobs, policy="distributed-backfill", num_nodes=2, node_gpus=8)
    assert sum(n.busy_gpu_s for n in res.nodes) == pytest.approx(sum(j.gpus * j.runtime_s for j in res.jobs))


def test_node_energy_apportioned_and_sums_to_cluster_total() -> None:
    # A 10-GPU job at 300 W/GPU tight-packs 8+2 on a 2x8 cluster. Node 0 hosts 8 GPUs and bills 8/10
    # of the energy, node 1 hosts 2 and bills 2/10; together they equal the cluster total.
    job = {"submit_s": 0, "gpus": 10, "duration_s": 10, "power_w_per_gpu": 300}
    res = schedule([job], policy="distributed-fcfs", num_nodes=2, node_gpus=8)
    total = res.cluster.total_energy_kwh
    assert res.jobs[0].nodes == ((0, 8), (1, 2))
    assert res.nodes[0].energy_kwh == pytest.approx(total * 8 / 10)
    assert res.nodes[1].energy_kwh == pytest.approx(total * 2 / 10)
    assert sum(n.energy_kwh for n in res.nodes) == pytest.approx(total)


def test_idle_node_has_no_energy() -> None:
    res = schedule(
        [{"submit_s": 0, "gpus": 2, "duration_s": 10, "power_w_per_gpu": 300}],
        policy="distributed-fcfs",
        num_nodes=2,
        node_gpus=8,
    )
    assert res.nodes[1].jobs_hosted == 0
    assert res.nodes[1].energy_kwh is None


def test_full_node_utilization_is_one() -> None:
    # One 8-GPU job holds node 0 fully for the whole (single-job) makespan: utilisation 1.0.
    res = schedule([{"submit_s": 0, "gpus": 8, "duration_s": 10}], policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    assert res.nodes[0].utilization == pytest.approx(1.0)
    assert res.nodes[0].idle_s == pytest.approx(0.0)
