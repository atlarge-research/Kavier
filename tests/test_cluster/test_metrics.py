"""Per-cluster metrics, energy, and the timeline step-series.

Every expected value is hand-derived from first principles in the comment beside it — never copied
from an engine run. Simple, fully-determined schedules are used so the arithmetic is checkable by eye.
"""

from __future__ import annotations

import pytest

from kavier.sdk.cluster import schedule


def test_energy_per_job_is_power_times_gpus_times_runtime() -> None:
    # 350 W/GPU x 2 GPUs x 10 s = 7000 W·s; kWh = 7000 / 3.6e6 = 0.0019444... kWh.
    res = schedule(
        [{"job_id": "a", "submit_s": 0, "gpus": 2, "duration_s": 10, "power_w_per_gpu": 350}],
        policy="distributed-fcfs",
        num_nodes=1,
        node_gpus=8,
    )
    assert res.jobs[0].energy_kwh == pytest.approx(7000 / 3.6e6)


def test_energy_falls_back_to_default_watts_when_job_has_no_power() -> None:
    # No per-job power; default 350 W/GPU used: 350 x 2 x 10 / 3.6e6 kWh.
    res = schedule(
        [{"job_id": "a", "submit_s": 0, "gpus": 2, "duration_s": 10}],
        policy="distributed-fcfs",
        num_nodes=1,
        node_gpus=8,
        default_watts_per_gpu=350,
    )
    assert res.jobs[0].energy_kwh == pytest.approx(350 * 2 * 10 / 3.6e6)


def test_energy_is_none_without_any_power_source() -> None:
    # No per-job power and no default -> energy is unknowable, not zero.
    res = schedule([{"submit_s": 0, "gpus": 2, "duration_s": 10}], policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    assert res.jobs[0].energy_kwh is None
    assert res.cluster.total_energy_kwh is None


def test_total_energy_is_sum_of_job_energy() -> None:
    # Two identical jobs at 350 W/GPU: total = 2 x (350 x 2 x 10 / 3.6e6).
    jobs = [
        {"submit_s": 0, "gpus": 2, "duration_s": 10, "power_w_per_gpu": 350},
        {"submit_s": 0, "gpus": 2, "duration_s": 10, "power_w_per_gpu": 350},
    ]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=8)  # both co-run on 8 GPUs
    assert res.cluster.total_energy_kwh == pytest.approx(2 * (350 * 2 * 10 / 3.6e6))


def test_utilization_is_gpu_seconds_over_capacity_times_makespan() -> None:
    # One 4-GPU job for 10 s on an 8-GPU cluster: AUC = 4x10 = 40 GPU·s;
    # capacity x makespan = 8 x 10 = 80; utilization = 40/80 = 0.5.
    res = schedule([{"submit_s": 0, "gpus": 4, "duration_s": 10}], policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    assert res.cluster.utilization == pytest.approx(0.5)


def test_utilization_is_one_when_serialized_jobs_saturate_the_pool() -> None:
    # Two 4-GPU jobs on a 4-GPU pool serialize [0,10],[10,20]: AUC = 80, cap x makespan = 4 x 20 = 80,
    # utilization = 1.0 (the pool is busy the whole time).
    jobs = [{"submit_s": 0, "gpus": 4, "duration_s": 10}, {"submit_s": 0, "gpus": 4, "duration_s": 10}]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=4)
    assert res.cluster.utilization == pytest.approx(1.0)


def test_goodput_is_jobs_per_makespan_second() -> None:
    # Two jobs serialized over a 20 s makespan: goodput = 2 / 20 = 0.1 jobs/s.
    jobs = [{"submit_s": 0, "gpus": 4, "duration_s": 10}, {"submit_s": 0, "gpus": 4, "duration_s": 10}]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=4)
    assert res.cluster.goodput_jobs_per_s == pytest.approx(0.1)


def test_peak_gpus_is_max_concurrent_gpus() -> None:
    # Two 4-GPU jobs co-run on 8 GPUs -> 8 GPUs in use at the peak.
    jobs = [{"submit_s": 0, "gpus": 4, "duration_s": 10}, {"submit_s": 0, "gpus": 4, "duration_s": 10}]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    assert res.cluster.peak_gpus == 8


def test_peak_queue_is_max_jobs_waiting() -> None:
    # Two 4-GPU jobs on a 4-GPU pool: the second waits while the first runs -> peak queue depth 1.
    jobs = [{"submit_s": 0, "gpus": 4, "duration_s": 10}, {"submit_s": 0, "gpus": 4, "duration_s": 10}]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=4)
    assert res.cluster.peak_queue == 1


def test_timeline_starts_at_zero_and_drains_to_zero() -> None:
    # Invariant: the GPUs-in-use series is anchored at t=0 and returns to 0 once all jobs finish.
    jobs = [{"submit_s": 0, "gpus": 4, "duration_s": 10}, {"submit_s": 0, "gpus": 4, "duration_s": 10}]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    tl = res.timeline
    assert tl.times_s[0] == 0.0
    assert len(tl.times_s) == len(tl.gpus_in_use) == len(tl.queue_depth)
    assert len(tl.times_s) >= 2
    assert tl.gpus_in_use[-1] == 0
    assert max(tl.gpus_in_use) <= res.cluster.capacity_gpus
