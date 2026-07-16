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


def test_scheduling_goodput_is_one_when_no_job_waits() -> None:
    # Two 4-GPU jobs co-run on 8 GPUs, both start at submit -> each job's turnaround == its runtime,
    # so training time == wall-clock and scheduling goodput = Σrun/Σturn = 200/200 = 1.0.
    jobs = [{"submit_s": 0, "gpus": 4, "duration_s": 100}, {"submit_s": 0, "gpus": 4, "duration_s": 100}]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    assert res.cluster.scheduling_goodput == pytest.approx(1.0)


def test_scheduling_goodput_is_train_time_over_wall_clock_with_queueing() -> None:
    # A,B (4 GPUs, 100 s) fill the 8-GPU pool at t=0; C (8 GPUs, 50 s) waits until they finish at 100.
    #   A,B: run=100, turnaround=100 (no wait);  C: run=50, wait=100, turnaround=150.
    #   scheduling goodput = Σrun / Σturn = (100+100+50) / (100+100+150) = 250/350 = 0.714286.
    # Distinct in form from goodput_jobs_per_s (= 3 / makespan 150 = 0.02): this catches confusing
    # the scheduling-efficiency ratio with the jobs-per-second throughput.
    jobs = [
        {"job_id": "A", "submit_s": 0, "gpus": 4, "duration_s": 100},
        {"job_id": "B", "submit_s": 0, "gpus": 4, "duration_s": 100},
        {"job_id": "C", "submit_s": 0, "gpus": 8, "duration_s": 50},
    ]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    assert res.cluster.scheduling_goodput == pytest.approx(250 / 350)
    assert res.cluster.scheduling_goodput != pytest.approx(res.cluster.goodput_jobs_per_s)


def test_per_job_goodput_is_runtime_over_turnaround() -> None:
    # C runs 50 s after waiting 100 s -> turnaround 150 s, so its goodput = 50/150 = 1/3; the two
    # jobs that never queue have goodput 1.0.
    jobs = [
        {"job_id": "A", "submit_s": 0, "gpus": 4, "duration_s": 100},
        {"job_id": "B", "submit_s": 0, "gpus": 4, "duration_s": 100},
        {"job_id": "C", "submit_s": 0, "gpus": 8, "duration_s": 50},
    ]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    by_id = {j.job_id: j for j in res.jobs}
    assert by_id["C"].goodput == pytest.approx(1 / 3)
    assert by_id["A"].goodput == pytest.approx(1.0)


@pytest.mark.parametrize("policy", ["distributed-fcfs", "consolidated-fcfs"])
def test_scheduling_goodput_is_bounded_in_unit_interval(policy: str) -> None:
    # Invariant: runtime_s <= turnaround_s for every job (turnaround = wait + runtime, wait >= 0), so
    # the aggregate Σrun / Σturn is always in [0, 1] whatever the policy or queueing. Six 6-GPU jobs on
    # an 8-GPU cluster can never co-run (6+6 > 8), so they serialize with real waits -> the bound is
    # actually exercised, not trivially 1.0.
    jobs = [{"submit_s": i * 5, "gpus": 6, "duration_s": 40} for i in range(6)]
    res = schedule(jobs, policy=policy, num_nodes=1, node_gpus=8)
    assert 0.0 <= res.cluster.scheduling_goodput <= 1.0


def test_scheduling_goodput_is_zero_for_empty_schedule() -> None:
    # No jobs -> no training time and no wall-clock -> goodput is 0.0 by definition (not NaN, not a crash).
    res = schedule([], policy="distributed-fcfs", num_nodes=1, node_gpus=8)
    assert res.cluster.n_jobs == 0
    assert res.cluster.scheduling_goodput == 0.0


def test_scheduling_goodput_counts_only_scheduled_jobs_not_drops() -> None:
    # Documented semantics: a dropped (never-run) job carries no timing and is excluded from the ratio.
    # A 2-GPU job runs immediately (goodput 1.0) while a 64-GPU job is dropped on an 8-GPU cluster, so
    # scheduling_goodput = 1.0 over the single placed job and the drop is reported separately. (Guards
    # that dropped jobs are not silently counted as either 0 or a NaN that would poison the ratio.)
    jobs = [
        {"job_id": "ok", "submit_s": 0, "gpus": 2, "duration_s": 100},
        {"job_id": "big", "submit_s": 0, "gpus": 64, "duration_s": 100},
    ]
    res = schedule(jobs, policy="distributed-fcfs", num_nodes=1, node_gpus=8, oversized="drop")
    assert res.dropped == ["big"]
    assert res.cluster.n_jobs == 1
    assert res.cluster.scheduling_goodput == pytest.approx(1.0)


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
