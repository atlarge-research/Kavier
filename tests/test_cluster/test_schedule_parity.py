"""Parity: kavier.sdk.cluster.schedule must reproduce the frozen thesis schedulers.

The oracle is NOT a captured engine output. It is the two frozen capsule schedulers
(reproducibility-capsule gen_exp2.py::schedule = flat-pool strict FCFS, and
gen_exp4.py::schedule_backfill = node-aware backfill) run by hand on a small fixture; the
expected per-job wait/start/end below were derived from those functions' semantics and
cross-checked by executing the exact frozen code on this fixture. Any drift in the ported
kernel (wrong discipline, lost head-of-line blocking, broken backfill, wrong oversized
handling) flips at least one asserted cell.

Fixture (16 GPUs = 2 nodes x 8; durations in whole hours):
    J0 (0s,  8gpu, 2h, 1 node)   J1 (0s, 12gpu, 1h, 2 nodes)   J2 (0s, 8gpu, 1h, 1 node)
    J3 (0s, 32gpu, 1h, 2 nodes; OVERSIZED)   J4 (3600s, 4gpu, 1h, 1 node; late arrival)
"""

from __future__ import annotations

import pytest

from kavier.sdk.cluster import schedule

_JOBS = [
    {"job_id": "J0", "submit_s": 0, "gpus": 8, "duration_s": 7200, "nodes": 1},
    {"job_id": "J1", "submit_s": 0, "gpus": 12, "duration_s": 3600, "nodes": 2},
    {"job_id": "J2", "submit_s": 0, "gpus": 8, "duration_s": 3600, "nodes": 1},
    {"job_id": "J3", "submit_s": 0, "gpus": 32, "duration_s": 3600, "nodes": 2},
    {"job_id": "J4", "submit_s": 3600, "gpus": 4, "duration_s": 3600, "nodes": 1},
]


def _by_id(result: object) -> dict[str, object]:
    return {j.job_id: j for j in result.jobs}  # type: ignore[attr-defined]


# --- FCFS: flat-pool strict First-Come-First-Served, head-of-line blocking, cap oversized ---
# Expected (from gen_exp2.schedule on this fixture): head-of-line serialization pins each job
# behind the previous one's start; J3's 32 GPUs are capped to the 16-GPU cluster; J4's late
# arrival still waits behind the blocked queue.
_FCFS_EXPECTED = {
    #        wait_h, start_h, end_h
    "J0": (0.0, 0.0, 2.0),
    "J1": (2.0, 2.0, 3.0),
    "J2": (3.0, 3.0, 4.0),  # <- pinned behind J1 (no backfill)
    "J3": (4.0, 4.0, 5.0),  # <- 32 GPUs capped to 16
    "J4": (4.0, 5.0, 6.0),  # <- late arrival serialized to the tail
}


@pytest.mark.parametrize("job_id, expected", _FCFS_EXPECTED.items())
def test_fcfs_matches_frozen_capsule_schedule(job_id: str, expected: tuple[float, float, float]) -> None:
    jobs = _by_id(schedule(_JOBS, policy="fcfs", num_nodes=2, node_gpus=8))
    wait_h, start_h, end_h = expected
    job = jobs[job_id]
    assert job.wait_h == pytest.approx(wait_h)
    assert job.start_h == pytest.approx(start_h)
    assert job.end_h == pytest.approx(end_h)


def test_fcfs_cluster_metrics_match_frozen_capsule() -> None:
    # gen_exp2.schedule on the fixture: makespan = max(end)-min(start) = 6h;
    # avg wait = (0+2+3+4+4)/5 = 2.6h; avg run = (2+1+1+1+1)/5 = 1.2h.
    res = schedule(_JOBS, policy="fcfs", num_nodes=2, node_gpus=8)
    assert res.cluster.makespan_h == pytest.approx(6.0)
    assert res.cluster.avg_wait_h == pytest.approx(2.6)
    assert res.cluster.avg_run_h == pytest.approx(1.2)


# --- BACKFILL: node-aware num_nodes x node_gpus, aggressive backfill, cap oversized ---
# Expected (from gen_exp4.schedule_backfill on this fixture): small/late jobs (J2, J4)
# backfill past the head-of-line-blocked J1; J3 capped to 16 GPUs across both nodes.
_BACKFILL_EXPECTED = {
    #        wait_h, start_h, end_h, gpus_placed
    "J0": (0.0, 0.0, 2.0, 8),
    "J1": (2.0, 2.0, 3.0, 12),
    "J2": (0.0, 0.0, 1.0, 8),  # <- backfills ahead of blocked J1
    "J3": (3.0, 3.0, 4.0, 16),  # <- 32 GPUs capped to the 2x8 cluster
    "J4": (0.0, 1.0, 2.0, 4),  # <- late arrival backfills onto the freed node
}


@pytest.mark.parametrize("job_id, expected", _BACKFILL_EXPECTED.items())
def test_backfill_matches_frozen_capsule_schedule(job_id: str, expected: tuple[float, float, float, int]) -> None:
    jobs = _by_id(schedule(_JOBS, policy="backfill", num_nodes=2, node_gpus=8))
    wait_h, start_h, end_h, gpus = expected
    job = jobs[job_id]
    assert job.wait_h == pytest.approx(wait_h)
    assert job.start_h == pytest.approx(start_h)
    assert job.end_h == pytest.approx(end_h)
    assert job.gpus == gpus


def test_backfill_cluster_metrics_match_frozen_capsule() -> None:
    # gen_exp4.schedule_backfill on the fixture: makespan = 4h; avg wait = (0+2+0+3+0)/5 = 1.0h.
    res = schedule(_JOBS, policy="backfill", num_nodes=2, node_gpus=8)
    assert res.cluster.makespan_h == pytest.approx(4.0)
    assert res.cluster.avg_wait_h == pytest.approx(1.0)


def test_fcfs_and_backfill_diverge_on_the_blocked_small_job() -> None:
    # The discipline difference, pinned as a property: strict FCFS pins J2 behind the
    # head-of-line-blocked J1 (starts 3h); backfill lets J2 jump ahead (starts 0h).
    fcfs = _by_id(schedule(_JOBS, policy="fcfs", num_nodes=2, node_gpus=8))
    backfill = _by_id(schedule(_JOBS, policy="backfill", num_nodes=2, node_gpus=8))
    assert fcfs["J2"].start_h == pytest.approx(3.0)
    assert backfill["J2"].start_h == pytest.approx(0.0)


# Tight-pack node placements on the 2x8 datacenter (hand-derived, most-free-first, id tiebreak):
_FCFS_NODES = {
    "J0": ((0, 8),),
    "J1": ((0, 8), (1, 4)),
    "J2": ((0, 8),),
    "J3": ((0, 8), (1, 8)),
    "J4": ((0, 4),),
}
_BACKFILL_NODES = {
    "J0": ((0, 8),),
    "J2": ((1, 8),),
    "J4": ((1, 4),),
    "J1": ((0, 8), (1, 4)),
    "J3": ((0, 8), (1, 8)),
}


@pytest.mark.parametrize("job_id, nodes", _FCFS_NODES.items())
def test_fcfs_node_assignment_is_tight_packed(job_id: str, nodes: tuple[tuple[int, int], ...]) -> None:
    jobs = _by_id(schedule(_JOBS, policy="fcfs", num_nodes=2, node_gpus=8))
    assert jobs[job_id].nodes == nodes


@pytest.mark.parametrize("job_id, nodes", _BACKFILL_NODES.items())
def test_backfill_node_assignment_is_tight_packed(job_id: str, nodes: tuple[tuple[int, int], ...]) -> None:
    jobs = _by_id(schedule(_JOBS, policy="backfill", num_nodes=2, node_gpus=8))
    assert jobs[job_id].nodes == nodes
