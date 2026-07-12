"""Public contract of ``schedule``: input formats, oversized handling, capacity resolution, errors."""

from __future__ import annotations

import pytest

from kavier.sdk.cluster import schedule


def test_accepts_tuple_jobs() -> None:
    # (submit_s, gpus, duration_s) tuples: two 4-GPU jobs co-run on 8 GPUs, both start at 0.
    res = schedule([(0, 4, 10), (0, 4, 10)], policy="fcfs", num_gpus=8)
    assert [j.start_s for j in res.jobs] == [0.0, 0.0]
    assert res.cluster.n_jobs == 2


def test_dataframe_and_dict_inputs_agree() -> None:
    pd = pytest.importorskip("pandas")
    rows = [
        {"submit_s": 0, "gpus": 4, "duration_s": 10},
        {"submit_s": 0, "gpus": 4, "duration_s": 10},
    ]
    from_dicts = schedule(rows, policy="fcfs", num_gpus=4)
    from_df = schedule(pd.DataFrame(rows), policy="fcfs", num_gpus=4)
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
    res = schedule(jobs, policy="fcfs", num_gpus=4, oversized="drop")
    assert res.cluster.n_jobs == 1
    assert [j.job_id for j in res.jobs] == ["ok"]
    assert res.dropped == ["big"]


def test_oversized_cap_clamps_to_capacity() -> None:
    # cap semantics (the frozen default): a 32-GPU request on a 16-GPU pool runs on 16.
    res = schedule([{"submit_s": 0, "gpus": 32, "duration_s": 10}], policy="fcfs", num_gpus=16)
    assert res.jobs[0].gpus == 16
    assert res.dropped == []


def test_backfill_caps_by_node_geometry_not_total() -> None:
    # A 16-GPU job asking for 1 node on a 2x8 cluster is capped by the node, not the total:
    # per_node = min(ceil(16/1), 8) = 8, placed on 1 node => 8 GPUs (half the cluster), not 16.
    job = {"submit_s": 0, "gpus": 16, "duration_s": 10, "nodes": 1}
    res = schedule([job], policy="backfill", num_nodes=2, node_gpus=8)
    assert res.jobs[0].gpus == 8


def test_backfill_from_num_gpus_defaults_to_a_single_node() -> None:
    # Convenience: backfill given only num_gpus treats it as one node of that size.
    res = schedule([{"submit_s": 0, "gpus": 4, "duration_s": 10}], policy="backfill", num_gpus=8)
    assert res.cluster.capacity_gpus == 8
    assert res.jobs[0].gpus == 4


def test_empty_jobs_returns_zeroed_result() -> None:
    res = schedule([], policy="fcfs", num_gpus=8)
    assert res.cluster.n_jobs == 0
    assert res.cluster.makespan_s == 0.0
    assert res.jobs == []
    assert res.timeline.times_s == []


def test_nan_power_is_treated_as_missing_not_poisoned() -> None:
    # A blank/NaN per-GPU power is UNKNOWN, not zero and not NaN: that job's energy is None and it
    # must not poison the cluster total (a NaN total also serialises to invalid JSON via the CLI).
    jobs = [
        {"job_id": "known", "submit_s": 0, "gpus": 2, "duration_s": 10, "power_w_per_gpu": 350},
        {"job_id": "blank", "submit_s": 0, "gpus": 2, "duration_s": 10, "power_w_per_gpu": float("nan")},
    ]
    res = schedule(jobs, policy="fcfs", num_gpus=8)
    by_id = {j.job_id: j for j in res.jobs}
    assert by_id["blank"].energy_kwh is None
    assert by_id["known"].energy_kwh == pytest.approx(350 * 2 * 10 / 3.6e6)
    # total sums only the known job — a finite number, never NaN.
    assert res.cluster.total_energy_kwh == pytest.approx(350 * 2 * 10 / 3.6e6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy": "round_robin", "num_gpus": 8},  # unknown policy
        {"policy": "fcfs", "oversized": "queue", "num_gpus": 8},  # unknown oversized mode
        {"policy": "fcfs"},  # no capacity given
        {"policy": "backfill"},  # no node geometry given
    ],
)
def test_invalid_arguments_raise_value_error(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        schedule([{"submit_s": 0, "gpus": 1, "duration_s": 1}], **kwargs)  # type: ignore[arg-type]
