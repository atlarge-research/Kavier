"""Discrete-event cluster schedulers: strict FCFS / backfill, spread or consolidated placement.

Import-light (``heapq`` + ``math`` + the stdlib-only ``vocab`` enums): this is the simulation kernel,
so keep pandas/numpy and the spec library out of it. The *spread* kernels (:func:`run_fcfs`,
:func:`run_backfill`) are parity-checked against frozen reference schedulers by
``tests/test_cluster/test_schedule_parity.py`` — don't change their scheduling logic. The
*consolidated* kernels (:func:`run_fcfs_consolidated`, :func:`run_backfill_consolidated`) honour each
job's ``nodes`` request via :func:`place_consolidated` (gang placement: a ``gpus`` job asking for
``nodes`` replicas lands on exactly ``nodes`` distinct nodes, evenly split, never scattered wider).

Both kernels take a list of :class:`Job`\\ s and return one :class:`Placement` per scheduled job
(a job dropped for being oversized gets none). Times are in seconds; a started job runs to completion.
"""

from __future__ import annotations

import heapq
import math
from typing import NamedTuple

from kavier.sdk.cluster.vocab import Oversized, PlacementStrategy


class Job(NamedTuple):
    """A schedulable job. The ``nodes`` field is ignored by the spread kernels (:func:`run_fcfs`,
    :func:`run_backfill` place tight-pack) and honoured by the consolidated kernels
    (:func:`run_fcfs_consolidated`, :func:`run_backfill_consolidated`)."""

    idx: int  # position in the caller's job list (not ``index`` — that shadows tuple.index)
    submit_s: float
    gpus: int
    duration_s: float
    nodes: int
    dependencies: tuple[int, ...] = ()


def job_schedulable(job: Job, completed: set[int]) -> bool:
    """Return True when all of job's dependency indices are in the completed set."""
    return all(dep in completed for dep in job.dependencies)


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


def place_consolidated(
    free: list[int],
    gpus: int,
    nodes: int,
    node_gpus: int,
    spread: bool = False,
) -> list[tuple[int, int]] | None:
    """Consolidated (gang) placement: put ``gpus`` GPUs on EXACTLY ``n_eff`` distinct nodes.

    Honours a job's ``nodes`` request so a ``gpus``/``nodes`` job is co-located on that many nodes
    and never scattered wider (the fix for the split-placement bug). ``free`` is the per-node
    free-GPU count. The algorithm:

    1. ``n_eff = max(1, min(nodes, len(free)))`` — clamp the request to the nodes that exist.
    2. even-split ``gpus`` across ``n_eff`` nodes so the demands sum to EXACTLY ``gpus``
       (``base, rem = divmod(gpus, n_eff)``; the first ``rem`` nodes get one extra). For the
       divisible case every demand is ``gpus // nodes``.
    3. if any single node's share exceeds ``node_gpus`` the job cannot fit one replica per node →
       return ``None`` (infeasible; this is the oversized signal for the consolidated policies).
    4. assign each demand (largest first) to a DISTINCT node with ``free >= demand``. Node order
       depends on ``spread``: ``False`` (default) → tightest-fit (least-free node first, lowest id
       on ties) to keep roomy nodes open (bin-packing); ``True`` → most-free node first (lowest id
       on ties), mirroring the Kubernetes ``LeastAllocated`` scorer to spread load evenly.

    Returns ``[(node_id, gpus_on_node), ...]`` sorted by node id (summing to ``gpus``), ``[]`` for
    ``gpus <= 0``, or ``None`` if it does not fit right now. Does not mutate ``free``.
    """
    if gpus <= 0:
        return []
    n_eff = max(1, min(nodes, len(free)))
    base, rem = divmod(gpus, n_eff)
    demands = [base + (1 if i < rem else 0) for i in range(n_eff)]
    if max(demands) > node_gpus:
        return None
    if spread:
        order = sorted(range(len(free)), key=lambda n: (-free[n], n))  # most-free first, id tiebreak
    else:
        order = sorted(range(len(free)), key=lambda n: (free[n], n))   # least-free first, id tiebreak
    used: set[int] = set()
    taken: dict[int, int] = {}
    for demand in sorted(demands, reverse=True):  # place the biggest replicas first
        chosen: int | None = None
        for node_id in order:
            if node_id not in used and free[node_id] >= demand:
                chosen = node_id
                break
        if chosen is None:  # no fresh distinct node can hold this replica's share
            return None
        used.add(chosen)
        taken[chosen] = demand
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

    job_by_idx = {job.idx: job for job in jobs}
    order = sorted(active, key=lambda t: t[1])  # by submission time; stable => FIFO on ties
    running: list[tuple[float, int, int]] = []  # min-heap of (end_s, idx, gpus)
    free = capacity_gpus
    completed: set[int] = set()
    last_start = order[0][1]
    scheduled: list[tuple[int, float, int, float]] = []  # (idx, start, gpus, duration)
    for index, submit, gpus, duration in order:
        start = max(submit, last_start)  # FCFS: never start before the previous job
        # release all jobs that finish by `start`
        while running and running[0][0] <= start:
            end_s, finished_idx, freed = heapq.heappop(running)
            free += freed
            completed.add(finished_idx)
        # wait for resources AND dependency satisfaction
        while free < gpus or not job_schedulable(job_by_idx[index], completed):
            end_s, finished_idx, freed = heapq.heappop(running)
            start = max(start, end_s)
            free += freed
            completed.add(finished_idx)
        free -= gpus
        heapq.heappush(running, (start + duration, index, gpus))
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

    job_by_idx = {job.idx: job for job in jobs}
    arrivals = sorted(prepared, key=lambda t: t[1])  # by submission time; stable => FIFO on ties
    n = len(arrivals)
    pending: list[tuple[int, float, int, float]] = []
    running: list[tuple[float, int, tuple[tuple[int, int], ...]]] = []  # min-heap of (end_s, idx, node_assignment)
    free = [node_gpus] * num_nodes
    completed: set[int] = set()
    next_arrival = 0
    done: dict[int, Placement] = {}

    time = arrivals[0][1]
    while len(done) < n:
        while next_arrival < n and arrivals[next_arrival][1] <= time:
            pending.append(arrivals[next_arrival])
            next_arrival += 1
        while running and running[0][0] <= time:
            _, finished_idx, freed = heapq.heappop(running)
            for node_id, count in freed:
                free[node_id] += count
            completed.add(finished_idx)
        admitted: list[int] = []
        for queue_pos, (index, _submit, gpus, duration) in enumerate(pending):
            if not job_schedulable(job_by_idx[index], completed):
                continue
            assigned = place(free, gpus)
            if assigned is None:
                continue
            for node_id, count in assigned:
                free[node_id] -= count
            nodes = tuple(assigned)
            heapq.heappush(running, (time + duration, index, nodes))
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


def _prepare_consolidated(
    jobs: list[Job], num_nodes: int, node_gpus: int, oversized: str
) -> list[tuple[int, float, int, int, float]]:
    """Apply consolidated oversized handling and return ``(idx, submit, gpus, nodes, duration)`` rows.

    A job is *infeasible* when even a fully-free cluster cannot host it consolidated — i.e. its
    per-node share exceeds ``node_gpus`` (which also covers "wants more GPUs than the whole cluster").
    ``oversized="drop"`` skips such a job (the facade reports it in ``dropped``). ``oversized="cap"``
    shrinks it so it fits while staying as consolidated as possible: clamp ``gpus`` to the cluster
    capacity, then widen ``nodes`` just enough that each node's even-split share is ``<= node_gpus``
    (bounded by ``num_nodes``). A feasible job is passed through unchanged.
    """
    capacity = num_nodes * node_gpus
    prepared: list[tuple[int, float, int, int, float]] = []
    for job in jobs:
        gpus = job.gpus
        nodes = max(1, int(job.nodes))
        if place_consolidated([node_gpus] * num_nodes, gpus, nodes, node_gpus) is None:
            if oversized == Oversized.DROP:
                continue
            gpus = min(gpus, capacity)  # cap: keep it as consolidated as it can be
            nodes = min(max(nodes, math.ceil(gpus / node_gpus)), num_nodes)
        prepared.append((job.idx, job.submit_s, gpus, nodes, job.duration_s))
    return prepared


def run_fcfs_consolidated(
    jobs: list[Job], num_nodes: int, node_gpus: int, oversized: str = "cap", spread: bool = False
) -> list[Placement]:
    """Strict FCFS timing with consolidated (gang) placement honouring each job's ``nodes`` request.

    Like :func:`run_fcfs` this is head-of-line: jobs start in submission order and never before the
    previous one (``start = max(submit, last_start)``). Unlike the flat-pool spread kernel, admission
    is node-aware — a job starts at the earliest time its consolidated placement
    (:func:`place_consolidated`) fits the per-node free GPUs, waiting for running jobs to finish if
    needed. ``oversized`` is handled by :func:`_prepare_consolidated` (``"drop"`` skips an infeasible
    job; ``"cap"`` shrinks it). ``spread`` selects the node-order strategy: ``False`` (default) →
    bin-packing (least-free first); ``True`` → ``LeastAllocated``-style (most-free first).
    """
    prepared = _prepare_consolidated(jobs, num_nodes, node_gpus, oversized)
    if not prepared:
        return []

    job_by_idx = {job.idx: job for job in jobs}
    order = sorted(prepared, key=lambda t: t[1])  # by submission time; stable => FIFO on ties
    free = [node_gpus] * num_nodes
    running: list[tuple[float, int, tuple[tuple[int, int], ...]]] = []  # min-heap of (end_s, idx, assignment)
    completed: set[int] = set()
    last_start = order[0][1]
    placements: list[Placement] = []
    for index, submit, gpus, nodes, duration in order:
        start = max(submit, last_start)  # FCFS: never start before the previous job
        # release all jobs that finish by `start`
        while running and running[0][0] <= start:
            _, finished_idx, freed = heapq.heappop(running)
            for node_id, count in freed:
                free[node_id] += count
            completed.add(finished_idx)
        assigned = place_consolidated(free, gpus, nodes, node_gpus, spread=spread)
        # advance time until consolidated placement fits AND dependencies are satisfied
        while assigned is None or not job_schedulable(job_by_idx[index], completed):
            if not running:  # pragma: no cover - _prepare_consolidated guarantees feasibility
                raise RuntimeError(f"consolidated placement infeasible for job {index}")
            end_s, finished_idx, freed = heapq.heappop(running)
            start = max(start, end_s)
            for node_id, count in freed:
                free[node_id] += count
            completed.add(finished_idx)
            assigned = place_consolidated(free, gpus, nodes, node_gpus, spread=spread)
        for node_id, count in assigned:
            free[node_id] -= count
        nodes_assignment = tuple(assigned)
        heapq.heappush(running, (start + duration, index, nodes_assignment))
        placements.append(Placement(index, start, gpus, nodes_assignment))
        last_start = start
    return placements


def run_backfill_consolidated(
    jobs: list[Job], node_gpus: int, num_nodes: int, oversized: str = "cap", spread: bool = False
) -> list[Placement]:
    """Best-effort FIFO + aggressive backfill with consolidated (gang) placement.

    Same event loop as :func:`run_backfill` — every tick, any queued job whose placement fits starts
    now, so a small later job backfills past a blocked larger one — but placement is
    :func:`place_consolidated`, so a ``gpus``/``nodes`` job lands on exactly that many distinct nodes
    (co-located) instead of tight-packed across fragments. ``oversized`` is handled by
    :func:`_prepare_consolidated` (``"drop"`` skips an infeasible job; ``"cap"`` shrinks it).
    ``spread`` selects the node-order strategy: ``False`` (default) → bin-packing (least-free first);
    ``True`` → ``LeastAllocated``-style (most-free first).
    """
    prepared = _prepare_consolidated(jobs, num_nodes, node_gpus, oversized)
    if not prepared:
        return []

    job_by_idx = {job.idx: job for job in jobs}
    arrivals = sorted(prepared, key=lambda t: t[1])  # by submission time; stable => FIFO on ties
    n = len(arrivals)
    pending: list[tuple[int, float, int, int, float]] = []
    running: list[tuple[float, int, tuple[tuple[int, int], ...]]] = []  # min-heap of (end_s, idx, assignment)
    free = [node_gpus] * num_nodes
    completed: set[int] = set()
    next_arrival = 0
    done: dict[int, Placement] = {}

    time = arrivals[0][1]
    while len(done) < n:
        while next_arrival < n and arrivals[next_arrival][1] <= time:
            pending.append(arrivals[next_arrival])
            next_arrival += 1
        while running and running[0][0] <= time:
            _, finished_idx, freed = heapq.heappop(running)
            for node_id, count in freed:
                free[node_id] += count
            completed.add(finished_idx)
        admitted: list[int] = []
        for queue_pos, (index, _submit, gpus, nodes, duration) in enumerate(pending):
            if not job_schedulable(job_by_idx[index], completed):
                continue
            assigned = place_consolidated(free, gpus, nodes, node_gpus, spread=spread)
            if assigned is None:
                continue
            for node_id, count in assigned:
                free[node_id] -= count
            nodes_assignment = tuple(assigned)
            heapq.heappush(running, (time + duration, index, nodes_assignment))
            done[index] = Placement(index, time, gpus, nodes_assignment)
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
