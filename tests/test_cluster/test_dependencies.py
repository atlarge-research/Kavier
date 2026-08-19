"""Dependency-chain scheduling: A→B→C end-to-end across all four policies, plus validation errors."""

from __future__ import annotations

import pytest

from kavier.sdk.cluster import schedule
from kavier.sdk.cluster.facade import _validate_dependencies, _normalise

_ALL_POLICIES = [
    "distributed-fcfs",
    "distributed-backfill",
    "consolidated-fcfs",
    "consolidated-backfill",
]


@pytest.mark.parametrize("policy", _ALL_POLICIES)
def test_abc_chain_end_times(policy: str) -> None:
    """A finishes at 10, B at 20, C at 30 — regardless of scheduling policy."""
    # A→B→C: each job takes 10 s on 1 GPU; single node with 8 GPUs.
    # A has no deps, B depends on A, C depends on B and A.
    # With 1 GPU used per job the cluster is never resource-constrained — the only thing that delays
    # B and C is their dependency on the previous job finishing.
    with_dependencies = [
        {"job_id": "A", "submit_s": 0, "gpus": 1, "duration_s": 10},
        {"job_id": "B", "submit_s": 1, "gpus": 1, "duration_s": 10, "dependencies": '["A"]'},
        {"job_id": "C", "submit_s": 2, "gpus": 1, "duration_s": 10, "dependencies": '["B", "A"]'},
    ]
    result = schedule(with_dependencies, policy=policy, num_nodes=1, node_gpus=8)
    by_id = {j.job_id: j for j in result.jobs}
    # End times: A finishes at 10, B can only start after A (ready at 10) → ends at 20,
    # C depends on both A and B (ready at 20) → ends at 30.
    assert by_id["A"].end_s == pytest.approx(10.0)
    assert by_id["B"].end_s == pytest.approx(20.0)
    assert by_id["C"].end_s == pytest.approx(30.0)
    # wait_s and turnaround_s are measured from ready_s = max(submit_s, max dep end_s).
    # A: no deps → ready_s=0, start=0 → wait=0, turnaround=10.
    assert by_id["A"].wait_s == pytest.approx(0.0)
    assert by_id["A"].turnaround_s == pytest.approx(10.0)
    # B: deps=[A], ready_s=max(1, 10)=10, start=10 → wait=0, turnaround=10.
    assert by_id["B"].wait_s == pytest.approx(0.0)
    assert by_id["B"].turnaround_s == pytest.approx(10.0)
    # C: deps=[B,A], ready_s=max(2, 20, 10)=20, start=20 → wait=0, turnaround=10.
    assert by_id["C"].wait_s == pytest.approx(0.0)
    assert by_id["C"].turnaround_s == pytest.approx(10.0)


@pytest.mark.parametrize("policy", _ALL_POLICIES)
def test_end_times_no_dependencies(policy: str) -> None:
    """A finishes at 10, B at 11, C at 12 — regardless of scheduling policy."""
    # A→B→C: each job takes 10 s on 1 GPU; single node with 8 GPUs.
    # None of the jobs have dependencies to other jobs.
    # With 1 GPU used per job the cluster is never resource-constrained,
    # every job will start the moment shortly after getting created.
    without_dependencies = [
        {"job_id": "A", "submit_s": 0, "gpus": 1, "duration_s": 10},
        {"job_id": "B", "submit_s": 1, "gpus": 1, "duration_s": 10},
        {"job_id": "C", "submit_s": 2, "gpus": 1, "duration_s": 10},
    ]
    result = schedule(without_dependencies, policy=policy, num_nodes=1, node_gpus=8)
    by_id = {j.job_id: j for j in result.jobs}
    assert by_id["A"].end_s == pytest.approx(10.0)
    assert by_id["B"].end_s == pytest.approx(11.0)
    assert by_id["C"].end_s == pytest.approx(12.0)

    # A: start=0 → wait=0, turnaround=10.
    assert by_id["A"].wait_s == pytest.approx(0.0)
    assert by_id["A"].turnaround_s == pytest.approx(10.0)
    # B: start=1 → wait=0, turnaround=10.
    assert by_id["B"].wait_s == pytest.approx(0.0)
    assert by_id["B"].turnaround_s == pytest.approx(10.0)
    # C: start=2 → wait=0, turnaround=10.
    assert by_id["C"].wait_s == pytest.approx(0.0)
    assert by_id["C"].turnaround_s == pytest.approx(10.0)

@pytest.mark.parametrize("policy", _ALL_POLICIES)
def test_end_times_no_dependencies_with_wait_time(policy: str) -> None:
    with_dependencies = [
        {"job_id": "A", "submit_s": 0, "gpus": 1, "duration_s": 10},
        {"job_id": "B", "submit_s": 1, "gpus": 1, "duration_s": 10},
    ]
    result = schedule(with_dependencies, policy=policy, num_nodes=1, node_gpus=1)
    by_id = {j.job_id: j for j in result.jobs}
    assert by_id["A"].end_s == pytest.approx(10.0)
    assert by_id["B"].end_s == pytest.approx(20.0)

    # A: start=0 → wait=0, turnaround=10.
    assert by_id["A"].wait_s == pytest.approx(0.0)
    assert by_id["A"].turnaround_s == pytest.approx(10.0)
    # B: start=10 → wait=9, turnaround=19.
    assert by_id["B"].wait_s == pytest.approx(9.0)
    assert by_id["B"].turnaround_s == pytest.approx(19.0)


# ---------------------------------------------------------------------------
# _validate_dependencies error cases
# ---------------------------------------------------------------------------

def test_validate_unknown_dependency_raises() -> None:
    jobs = [
        {"job_id": "A", "submit_s": 0, "gpus": 1, "duration_s": 10, "dependencies": '["DOES_NOT_EXIST"]'},
    ]
    norm = _normalise(jobs)
    with pytest.raises(ValueError, match="not a known job_id"):
        _validate_dependencies(norm)


def test_validate_same_submit_time_raises() -> None:
    """A dependency submitted at the same time as the depending job is not strictly earlier."""
    jobs = [
        {"job_id": "X", "submit_s": 5, "gpus": 1, "duration_s": 10},
        {"job_id": "Y", "submit_s": 5, "gpus": 1, "duration_s": 10, "dependencies": '["X"]'},
    ]
    norm = _normalise(jobs)
    with pytest.raises(ValueError, match="strictly earlier submit_s"):
        _validate_dependencies(norm)


def test_validate_later_dependency_raises() -> None:
    """A dependency submitted after the depending job is also invalid."""
    jobs = [
        {"job_id": "late", "submit_s": 10, "gpus": 1, "duration_s": 5},
        {"job_id": "early", "submit_s": 1, "gpus": 1, "duration_s": 5, "dependencies": '["late"]'},
    ]
    norm = _normalise(jobs)
    with pytest.raises(ValueError, match="strictly earlier submit_s"):
        _validate_dependencies(norm)


def test_validate_no_error_for_empty_dependencies() -> None:
    jobs = [
        {"job_id": "A", "submit_s": 0, "gpus": 1, "duration_s": 10},
        {"job_id": "B", "submit_s": 5, "gpus": 1, "duration_s": 10},
    ]
    norm = _normalise(jobs)
    _validate_dependencies(norm)  # must not raise
