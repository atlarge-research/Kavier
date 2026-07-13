"""Discrete-event cluster schedulers: strict FCFS (flat pool) and node-aware backfill.

Import-light (``heapq`` + the stdlib-only ``vocab`` enums): this is the simulation kernel, so keep
pandas/numpy and the spec library out of it. Behaviour is parity-checked against frozen reference schedulers by
``tests/test_cluster/test_schedule_parity.py`` — don't change the scheduling logic.

Both kernels take a list of :class:`Job`\\ s and return one :class:`Placement` per scheduled job
(a job dropped for being oversized gets none). Times are in seconds; a started job runs to completion.
"""

from __future__ import annotations

import heapq
from typing import NamedTuple

from kavier.sdk.cluster.vocab import Oversized


class Job(NamedTuple):
    """A schedulable job. The ``nodes`` field is currently ignored by both kernels (placement is
    automatic tight-pack); it is reserved for future placement constraints."""

    idx: int  # position in the caller's job list (not ``index`` — that shadows tuple.index)
    submit_s: float
    gpus: int
    duration_s: float
    nodes: int


class Placement(NamedTuple):
    """The scheduler's decision for one job: start time, GPU count, and per-node assignment."""

    idx: int
    start_s: float
    gpus: int
    nodes: tuple[tuple[int, int], ...]  # ((node_id, gpus_on_node), ...); sum of gpus_on_node == gpus


def place(free: list[int], gpus: int) -> list[tuple[int, int]] | None:
    """Tight-pack ``gpus`` GPUs, best-fit: fill the least-free node first ("8+2").

    ``free`` is the per-node free-GPU count. Fills the least-free node first (ties broken by lowest
    node id), allowing partial nodes, until ``gpus`` are placed — consolidating onto small gaps and
    keeping the roomiest nodes open. Returns the assignment as ``[(node_id, gpus_on_node), ...]``
    sorted by node id, or ``None`` if ``sum(free) < gpus`` (the job does not fit). ``gpus <= 0``
    returns ``[]``. Does not mutate ``free``.
    """
    if gpus <= 0:
        return []
    if sum(free) < gpus:
        return None
    remaining = gpus
    taken: dict[int, int] = {}
    for node_id in sorted(range(len(free)), key=lambda n: (free[n], n)):
        if remaining <= 0:
            break
        avail = free[node_id]
        if avail <= 0:
            continue
        take = avail if avail < remaining else remaining
        taken[node_id] = take
        remaining -= take
    return sorted(taken.items())


def run_fcfs(jobs: list[Job], num_nodes: int, node_gpus: int, oversized: str = "cap") -> list[Placement]:
    """Strict First-Come-First-Served timing on a flat pool of ``num_nodes * node_gpus`` GPUs.

    The timing (start/end of every job) is the frozen ``gen_exp2.py::schedule`` behaviour — jobs run
    in submission order and never start before the previous one (head-of-line blocking). Node IDs are
    assigned afterwards by :func:`_assign_nodes` (tight-pack), which never changes the timing.
    ``oversized="cap"`` clamps a job wanting more than the whole cluster; ``"drop"`` skips it.
    """
    capacity_gpus = num_nodes * node_gpus
    active: list[tuple[int, float, int, float]] = []
    for job in jobs:
        gpus = job.gpus
        if gpus > capacity_gpus:
            if oversized == Oversized.DROP:
                continue
            gpus = capacity_gpus  # cap
        active.append((job.idx, job.submit_s, gpus, job.duration_s))
    if not active:
        return []

    order = sorted(active, key=lambda t: t[1])  # by submission time; stable => FIFO on ties
    running: list[tuple[float, int]] = []  # min-heap of (end_s, gpus)
    free = capacity_gpus
    last_start = order[0][1]
    scheduled: list[tuple[int, float, int, float]] = []  # (idx, start, gpus, duration)
    for index, submit, gpus, duration in order:
        start = max(submit, last_start)  # FCFS: never start before the previous job
        while free < gpus:
            end_s, freed = heapq.heappop(running)
            start = max(start, end_s)
            free += freed
        free -= gpus
        heapq.heappush(running, (start + duration, gpus))
        scheduled.append((index, start, gpus, duration))
        last_start = start

    assignments = _assign_nodes(scheduled, num_nodes, node_gpus)
    return [Placement(idx, start, gpus, assignments[idx]) for idx, start, gpus, _ in scheduled]


def _assign_nodes(
    scheduled: list[tuple[int, float, int, float]], num_nodes: int, node_gpus: int
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Post-hoc node placement for an already-timed schedule (used by FCFS).

    Replays ``scheduled`` (``(idx, start, gpus, duration)``) in ``(start, idx)`` order on a fresh
    ``num_nodes x node_gpus`` datacenter, tight-packing each job with :func:`place`. Any schedule a
    flat pool of ``num_nodes*node_gpus`` GPUs produced is node-feasible under tight-pack, so
    :func:`place` never returns ``None`` here.
    """
    free = [node_gpus] * num_nodes
    running: list[tuple[float, tuple[tuple[int, int], ...]]] = []  # (end_s, node_assignment)
    assignments: dict[int, tuple[tuple[int, int], ...]] = {}
    for idx, start, gpus, duration in sorted(scheduled, key=lambda s: (s[1], s[0])):
        while running and running[0][0] <= start:
            _, freed = heapq.heappop(running)
            for node_id, count in freed:
                free[node_id] += count
        assigned = place(free, gpus)
        if assigned is None:  # pragma: no cover - flat-pool feasibility guarantees a fit
            raise RuntimeError(f"node placement infeasible for job {idx}")
        for node_id, count in assigned:
            free[node_id] -= count
        nodes = tuple(assigned)
        heapq.heappush(running, (start + duration, nodes))
        assignments[idx] = nodes
    return assignments


def run_backfill(jobs: list[Job], node_gpus: int, num_nodes: int, oversized: str = "cap") -> list[Placement]:
    """Best-effort FIFO with aggressive backfill on a ``num_nodes x node_gpus`` cluster.

    Jobs are considered in submission order every tick; any queued job that fits (tight-pack,
    ``sum(free) >= gpus``) starts now, so a small later job backfills past a larger blocked one. Node
    IDs are assigned intrinsically by :func:`place`. A job wanting more than the whole cluster is
    capped to the total (``oversized="cap"``) or skipped (``"drop"``). The per-job ``nodes`` request
    field is ignored — placement is automatic.
    """
    total = node_gpus * num_nodes
    prepared: list[tuple[int, float, int, float]] = []  # (idx, submit, gpus, duration)
    for job in jobs:
        gpus = job.gpus
        if gpus > total:
            if oversized == Oversized.DROP:
                continue
            gpus = total  # cap
        prepared.append((job.idx, job.submit_s, gpus, job.duration_s))
    if not prepared:
        return []

    arrivals = sorted(prepared, key=lambda t: t[1])  # by submission time; stable => FIFO on ties
    n = len(arrivals)
    pending: list[tuple[int, float, int, float]] = []
    running: list[tuple[float, tuple[tuple[int, int], ...]]] = []  # min-heap of (end_s, node_assignment)
    free = [node_gpus] * num_nodes
    next_arrival = 0
    done: dict[int, Placement] = {}

    time = arrivals[0][1]
    while len(done) < n:
        while next_arrival < n and arrivals[next_arrival][1] <= time:
            pending.append(arrivals[next_arrival])
            next_arrival += 1
        while running and running[0][0] <= time:
            _, freed = heapq.heappop(running)
            for node_id, count in freed:
                free[node_id] += count
        admitted: list[int] = []
        for queue_pos, (index, _submit, gpus, duration) in enumerate(pending):
            assigned = place(free, gpus)
            if assigned is None:
                continue
            for node_id, count in assigned:
                free[node_id] -= count
            nodes = tuple(assigned)
            heapq.heappush(running, (time + duration, nodes))
            done[index] = Placement(index, time, gpus, nodes)
            admitted.append(queue_pos)
        for queue_pos in reversed(admitted):
            pending.pop(queue_pos)
        candidates: list[float] = []
        if next_arrival < n:
            candidates.append(arrivals[next_arrival][1])
        if running:
            candidates.append(running[0][0])
        if not candidates:
            break
        time = max(time, min(candidates))
    return [done[i] for i in sorted(done)]
