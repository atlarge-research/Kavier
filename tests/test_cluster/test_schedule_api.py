"""Public contract of ``schedule``: input formats, oversized handling, capacity resolution, errors."""

from __future__ import annotations

import pytest

from kavier.sdk.cluster import schedule


def test_accepts_tuple_jobs() -> None:
    # (submit_s, gpus, duration_s) tuples: two 4-GPU jobs co-run on 8 GPUs, both start at 0.
    res = schedule([(0, 4, 10), (0, 4, 10)], policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    assert [j.start_s for j in res.jobs] == [0.0, 0.0]
    assert res.cluster.n_jobs == 2


def test_dataframe_and_dict_inputs_agree() -> None:
    pd = pytest.importorskip("pandas")
    rows = [
        {"submit_s": 0, "gpus": 4, "duration_s": 10},
        {"submit_s": 0, "gpus": 4, "duration_s": 10},
    ]
    from_dicts = schedule(rows, policy="distributed-fcfs", num_nodes=1, node_gpus=4)
    from_df = schedule(pd.DataFrame(rows), policy="distributed-fcfs", num_nodes=1, node_gpus=4)
    # Same schedule regardless of container: serialized [0,10],[10,20] on the 4-GPU pool.
    assert [(j.start_s, j.end_s) for j in from_df.jobs] == [(j.start_s, j.end_s) for j in from_dicts.jobs]
    assert [(0.0, 10.0), (10.0, 20.0)] == [(j.start_s, j.end_s) for j in from_df.jobs]


def test_oversized_drop_excludes_the_job_and_reports_it() -> None:
    # simulate_fifo parity: a job wanting more GPUs than the whole cluster is dropped (would block
    # FIFO forever), the other survives.
    jobs = [
        {"job_id": "big", "submit_s": 0, "gpus": 999, "duration_s": 10},
        {"job_id": "ok", "submit_s": 0, "gpus": 2, "duration_s": 10},
    ]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=4, oversized="drop")
    assert res.cluster.n_jobs == 1
    assert [j.job_id for j in res.jobs] == ["ok"]
    assert res.dropped == ["big"]


def test_oversized_cap_clamps_to_capacity() -> None:
    # cap semantics (the frozen default): a 32-GPU request on a 16-GPU pool runs on 16.
    res = schedule([{"submit_s": 0, "gpus": 32, "duration_s": 10}], policy="distributed-fcfs", num_nodes=2, node_gpus=8)
    assert res.jobs[0].gpus == 16
    assert res.dropped == []


def test_backfill_tight_packs_across_nodes_no_gpu_dropped() -> None:
    # A 16-GPU job on a 2x8 cluster now fills both nodes (8+8) instead of being capped to one node's 8.
    job = {"submit_s": 0, "gpus": 16, "duration_s": 10}
    res = schedule([job], policy="distributed-backfill", num_nodes=2, node_gpus=8)
    assert res.jobs[0].gpus == 16
    assert res.jobs[0].nodes == ((0, 8), (1, 8))


def test_backfill_on_a_single_node() -> None:
    res = schedule(
        [{"submit_s": 0, "gpus": 4, "duration_s": 10}], policy="distributed-backfill", num_nodes=1, node_gpus=8
    )
    assert res.cluster.capacity_gpus == 8
    assert res.jobs[0].gpus == 4
    assert res.jobs[0].nodes == ((0, 4),)


def test_empty_jobs_returns_zeroed_result() -> None:
    res = schedule([], policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    assert res.cluster.n_jobs == 0
    assert res.cluster.makespan_s == 0.0
    assert res.jobs == []
    assert res.timeline.times_s == []
    assert len(res.nodes) == 1


def test_nan_power_is_treated_as_missing_not_poisoned() -> None:
    # A blank/NaN per-GPU power is UNKNOWN, not zero and not NaN: that job's energy is None and it
    # must not poison the cluster total (a NaN total also serialises to invalid JSON via the CLI).
    jobs = [
        {"job_id": "known", "submit_s": 0, "gpus": 2, "duration_s": 10, "power_w_per_gpu": 350},
        {"job_id": "blank", "submit_s": 0, "gpus": 2, "duration_s": 10, "power_w_per_gpu": float("nan")},
    ]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    by_id = {j.job_id: j for j in res.jobs}
    assert by_id["blank"].energy_kwh is None
    assert by_id["known"].energy_kwh == pytest.approx(350 * 2 * 10 / 3.6e6)
    # total sums only the known job — a finite number, never NaN.
    assert res.cluster.total_energy_kwh == pytest.approx(350 * 2 * 10 / 3.6e6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy": "round_robin", "num_nodes": 1, "node_gpus": 8},  # unknown policy
        {"policy": "distributed-fcfs", "oversized": "queue", "num_nodes": 1, "node_gpus": 8},  # unknown oversized mode
        {"policy": "distributed-fcfs", "placement": "random", "num_nodes": 1, "node_gpus": 8},  # unknown placement
        {"policy": "distributed-fcfs"},  # no topology given
        {"policy": "distributed-fcfs", "num_nodes": 2},  # node_gpus missing
        {"policy": "distributed-backfill", "node_gpus": 8},  # num_nodes missing
        {"policy": "distributed-fcfs", "num_nodes": 0, "node_gpus": 8},  # non-positive
    ],
)
def test_invalid_arguments_raise_value_error(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        schedule([{"submit_s": 0, "gpus": 1, "duration_s": 1}], **kwargs)  # type: ignore[arg-type]


def test_placement_spread_distributes_across_nodes() -> None:
    # Two sequential 1-node/4-GPU jobs on 2x8 cluster with spread: each lands on a different node.
    jobs = [
        {"job_id": "a", "submit_s": 0, "gpus": 4, "duration_s": 10, "nodes": 1},
        {"job_id": "b", "submit_s": 0, "gpus": 4, "duration_s": 10, "nodes": 1},
    ]
    res = schedule(jobs, policy="consolidated-backfill", num_nodes=2, node_gpus=8, placement="spread")
    by_id = {j.job_id: j for j in res.jobs}
    # Each job must be on a distinct node
    node_a = by_id["a"].nodes[0][0]
    node_b = by_id["b"].nodes[0][0]
    assert node_a != node_b, f"spread should place jobs on different nodes, got both on node {node_a}"


def test_placement_pack_consolidates_onto_same_node() -> None:
    # Same scenario with pack: both jobs land on node 0.
    jobs = [
        {"job_id": "a", "submit_s": 0, "gpus": 4, "duration_s": 10, "nodes": 1},
        {"job_id": "b", "submit_s": 0, "gpus": 4, "duration_s": 10, "nodes": 1},
    ]
    res = schedule(jobs, policy="consolidated-backfill", num_nodes=2, node_gpus=8, placement="pack")
    by_id = {j.job_id: j for j in res.jobs}
    node_a = by_id["a"].nodes[0][0]
    node_b = by_id["b"].nodes[0][0]
    assert node_a == node_b == 0, f"pack should place both jobs on node 0, got {node_a} and {node_b}"


def test_placement_default_is_pack() -> None:
    # Omitting placement= gives the same result as placement="pack".
    jobs = [
        {"job_id": "a", "submit_s": 0, "gpus": 4, "duration_s": 10, "nodes": 1},
        {"job_id": "b", "submit_s": 0, "gpus": 4, "duration_s": 10, "nodes": 1},
    ]
    explicit = schedule(jobs, policy="consolidated-backfill", num_nodes=2, node_gpus=8, placement="pack")
    default = schedule(jobs, policy="consolidated-backfill", num_nodes=2, node_gpus=8)
    assert [(j.job_id, j.nodes) for j in explicit.jobs] == [(j.job_id, j.nodes) for j in default.jobs]
