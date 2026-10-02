"""Public API of ``schedule``: input formats, oversized handling, capacity resolution, errors."""

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
    # As in simulate_fifo, a job wanting more GPUs than the cluster has is dropped, since it would block
    # FIFO forever. The other job runs.
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
    # A 16-GPU job on a 2x8 cluster fills both nodes as 8+8.
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
    # A blank or NaN per-GPU power means unknown: that job's energy is None and the cluster total stays
    # finite. A NaN total would also serialise to invalid JSON in the CLI.
    jobs = [
        {"job_id": "known", "submit_s": 0, "gpus": 2, "duration_s": 10, "power_w_per_gpu": 350},
        {"job_id": "blank", "submit_s": 0, "gpus": 2, "duration_s": 10, "power_w_per_gpu": float("nan")},
    ]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    by_id = {j.job_id: j for j in res.jobs}
    assert by_id["blank"].energy_kwh is None
    assert by_id["known"].energy_kwh == pytest.approx(350 * 2 * 10 / 3.6e6)
    # The total covers only the known job.
    assert res.cluster.total_energy_kwh == pytest.approx(350 * 2 * 10 / 3.6e6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy": "round_robin", "num_nodes": 1, "node_gpus": 8},  # unknown policy
        {"policy": "distributed-fcfs", "oversized": "queue", "num_nodes": 1, "node_gpus": 8},  # unknown oversized mode
        {"policy": "distributed-fcfs"},  # no topology given
        {"policy": "distributed-fcfs", "num_nodes": 2},  # node_gpus missing
        {"policy": "distributed-backfill", "node_gpus": 8},  # num_nodes missing
        {"policy": "distributed-fcfs", "num_nodes": 0, "node_gpus": 8},  # non-positive
    ],
)
def test_invalid_arguments_raise_value_error(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        schedule([{"submit_s": 0, "gpus": 1, "duration_s": 1}], **kwargs)  # type: ignore[arg-type]


def test_distributed_fcfs_zero_duration_job_tied_with_a_later_job() -> None:
    # B runs [0, 5]. Z (0 s long) waits for B and starts at 5; W is listed first but submitted at 1 s,
    # so it runs after Z and also starts at 5 on the GPUs Z frees. Node assignment used to replay W
    # before Z and raise RuntimeError.
    jobs = [
        {"job_id": "W", "submit_s": 1, "gpus": 8, "duration_s": 10},
        {"job_id": "B", "submit_s": 0, "gpus": 8, "duration_s": 5},
        {"job_id": "Z", "submit_s": 0, "gpus": 8, "duration_s": 0},
    ]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    by_id = {j.job_id: j for j in res.jobs}
    assert [by_id[k].start_s for k in ("B", "Z", "W")] == [0.0, 5.0, 5.0]
    assert [by_id[k].nodes for k in ("B", "Z", "W")] == [((0, 8),)] * 3


def test_dataframe_blank_nodes_cell_defaults_to_one_node() -> None:
    # A missing value in an optional nodes column reaches the facade as NaN and means one node.
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        [
            {"job_id": "a", "submit_s": 0, "gpus": 8, "duration_s": 10, "nodes": 2},
            {"job_id": "b", "submit_s": 0, "gpus": 8, "duration_s": 10, "nodes": None},
        ]
    )
    res = schedule(df, policy="consolidated-fcfs", num_nodes=4, node_gpus=8)
    by_id = {j.job_id: j for j in res.jobs}
    assert by_id["a"].nodes == ((0, 4), (1, 4))
    assert by_id["b"].nodes == ((2, 8),)


@pytest.mark.parametrize(
    "bad_job",
    [
        {"submit_s": float("inf"), "gpus": 1, "duration_s": 1},
        {"submit_s": 0, "gpus": 1, "duration_s": float("inf")},
        {"submit_s": 0, "gpus": 1, "duration_s": float("-inf")},
        {"submit_s": 0, "gpus": 1, "duration_s": -5},
        {"submit_s": 0, "gpus": -2, "duration_s": 5},
        {"submit_s": 0, "gpus": float("inf"), "duration_s": 5},
        {"submit_s": 0, "gpus": float("nan"), "duration_s": 5},
    ],
)
def test_infinite_or_negative_job_values_raise_value_error(bad_job: dict[str, float]) -> None:
    jobs = [{"submit_s": 0, "gpus": 1, "duration_s": 1}, bad_job]
    with pytest.raises(ValueError, match="job 1"):
        schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=8)


def test_zero_gpus_and_zero_duration_are_still_accepted() -> None:
    res = schedule([(0, 0, 10), (0, 2, 0)], policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    assert res.cluster.n_jobs == 2


def test_tuple_rows_read_power_and_job_id() -> None:
    # (submit_s, gpus, duration_s, nodes, power_w_per_gpu, job_id): 500 W x 2 GPUs x 1 h = 1 kWh.
    res = schedule([(0, 2, 3600, 1, 500, "a")], policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    assert res.jobs[0].job_id == "a"
    assert res.jobs[0].energy_kwh == pytest.approx(1.0)


def test_tuple_rows_without_job_id_use_the_row_index_and_nan_power_is_unknown() -> None:
    res = schedule(
        [(0, 2, 3600, 1, 500), (0, 2, 3600, 1, float("nan"))], policy="distributed-fcfs", num_nodes=1, node_gpus=8
    )
    assert [j.job_id for j in res.jobs] == [0, 1]
    assert res.jobs[0].energy_kwh == pytest.approx(1.0)
    assert res.jobs[1].energy_kwh is None
