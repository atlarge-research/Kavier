"""Discrete-event cluster schedulers: strict FCFS or backfill, with spread or consolidated placement.

Imports only ``heapq``, ``math`` and the stdlib-only ``vocab`` enums; keep pandas, numpy and the spec
library out of this kernel. The spread kernels (:func:`run_fcfs`, :func:`run_backfill`) are
parity-checked against frozen reference schedulers in ``tests/test_cluster/test_schedule_parity.py``,
so their scheduling logic must not change. The consolidated kernels (:func:`run_fcfs_consolidated`,
:func:`run_backfill_consolidated`) honour each job's ``nodes`` request via :func:`place_consolidated`:
a job lands on exactly ``nodes`` distinct nodes with an even GPU split.

Every kernel takes a list of :class:`Job` and returns one :class:`Placement` per scheduled job; a job
dropped as oversized gets none. Times are in seconds, and a started job runs to completion.
"""

from __future__ import annotations

import heapq
import math
from typing import NamedTuple

from kavier.sdk.cluster.vocab import Oversized


class Job(NamedTuple):
    """A schedulable job; ``nodes`` is ignored by the spread kernels and honoured by the consolidated ones."""

    idx: int  # position in the caller's job list; ``index`` would shadow tuple.index
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
    """Tight-pack ``gpus`` GPUs best-fit, filling the least-free node first ("8+2").

    ``free`` is the free-GPU count per node. Nodes fill in ``(free, node_id)`` order, partial nodes
    allowed, which packs small gaps and keeps the roomiest nodes open. Returns
    ``[(node_id, gpus_on_node), ...]`` sorted by node id, ``None`` if ``sum(free) < gpus``, or ``[]`` if
    ``gpus <= 0``. Does not mutate ``free``.
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


def place_consolidated(free: list[int], gpus: int, nodes: int, node_gpus: int) -> list[tuple[int, int]] | None:
    """Gang-place ``gpus`` GPUs on exactly ``n_eff`` distinct nodes.

    ``free`` is the free-GPU count per node. Steps:

    1. ``n_eff = max(1, min(nodes, len(free), gpus))`` clamps the request to the existing nodes and
       to one GPU per node, so no node gets a zero-GPU share.
    2. Split ``gpus`` evenly over ``n_eff`` nodes: ``base, rem = divmod(gpus, n_eff)``, and the first
       ``rem`` nodes get one extra, so the demands sum to ``gpus``.
    3. If one share exceeds ``node_gpus``, return ``None``. This is the oversized signal for the
       consolidated policies.
    4. Assign each demand, largest first, to a distinct node with ``free >= demand``, tightest fit
       (least-free node that fits, lowest id on ties), which keeps roomy nodes open.

    Returns ``[(node_id, gpus_on_node), ...]`` sorted by node id and summing to ``gpus``, ``[]`` for
    ``gpus <= 0``, or ``None`` if it does not fit now. Does not mutate ``free``.
    """
    if gpus <= 0:
        return []
    n_eff = max(1, min(nodes, len(free), gpus))
    base, rem = divmod(gpus, n_eff)
    demands = [base + (1 if i < rem else 0) for i in range(n_eff)]
    if max(demands) > node_gpus:
        return None
    order = sorted(range(len(free)), key=lambda n: (free[n], n))  # tightest-fit, id tiebreak
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
    """Schedule ``jobs`` strict first-come-first-served on a flat pool of ``num_nodes * node_gpus`` GPUs.

    Start and end times match the frozen ``gen_exp2.py::schedule``: jobs run in submission order and
    never start before the previous one (head-of-line blocking). :func:`_assign_nodes` then assigns
    node IDs by tight-pack without changing the timing. ``oversized="cap"`` clamps a job larger than
    the cluster; ``"drop"`` skips it.
    """
    capacity_gpus = num_nodes * node_gpus
    active: list[tuple[int, float, int, float]] = []
    for job in jobs:
        gpus = job.gpus
        if gpus > capacity_gpus:
            if oversized == Oversized.DROP:
                continue
            gpus = capacity_gpus
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
    """Assign nodes to an already-timed FCFS schedule.

    Replays ``scheduled`` (``(idx, start, gpus, duration)``) in its run order, where ``start`` never
    decreases, on an empty ``num_nodes x node_gpus`` cluster, tight-packing each job with
    :func:`place`. A schedule feasible on a flat pool of ``num_nodes * node_gpus`` GPUs is
    node-feasible under tight-pack, so :func:`place` never returns ``None`` here. Run order matters
    when a zero-duration job and a later job start at the same instant: the later job needs the GPUs
    the zero-duration job frees.
    """
    free = [node_gpus] * num_nodes
    running: list[tuple[float, tuple[tuple[int, int], ...]]] = []  # (end_s, node_assignment)
    assignments: dict[int, tuple[tuple[int, int], ...]] = {}
    for idx, start, gpus, duration in scheduled:
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
    """Schedule ``jobs`` FIFO with aggressive backfill on a ``num_nodes x node_gpus`` cluster.

    At each event time, queued jobs are tried in submission order. Any job that fits by tight-pack
    (``sum(free) >= gpus``) starts, so a small later job can pass a larger blocked one. :func:`place`
    assigns the nodes. A job larger than the cluster is capped (``oversized="cap"``) or skipped
    (``"drop"``). The per-job ``nodes`` request is ignored.
    """
    total = node_gpus * num_nodes
    prepared: list[tuple[int, float, int, float]] = []  # (idx, submit, gpus, duration)
    for job in jobs:
        gpus = job.gpus
        if gpus > total:
            if oversized == Oversized.DROP:
                continue
            gpus = total
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


def _prepare_consolidated(
    jobs: list[Job], num_nodes: int, node_gpus: int, oversized: str
) -> list[tuple[int, float, int, int, float]]:
    """Apply consolidated oversized handling; return ``(idx, submit, gpus, nodes, duration)`` rows.

    A job is infeasible when an empty cluster cannot host it consolidated, i.e. its per-node share
    exceeds ``node_gpus``. ``oversized="drop"`` skips an infeasible job only when it asks for more
    GPUs than the cluster has, and the facade lists it in ``dropped``. Every other infeasible job is
    handled as under ``oversized="cap"``: clamp ``gpus`` to cluster capacity, then widen ``nodes``
    until each even-split share is ``<= node_gpus``, up to ``num_nodes``. Feasible jobs pass through
    unchanged.
    """
    capacity = num_nodes * node_gpus
    prepared: list[tuple[int, float, int, int, float]] = []
    for job in jobs:
        gpus = job.gpus
        nodes = max(1, int(job.nodes))
        if place_consolidated([node_gpus] * num_nodes, gpus, nodes, node_gpus) is None:
            if oversized == Oversized.DROP and gpus > capacity:
                continue
            gpus = min(gpus, capacity)  # cap, widening nodes as little as possible
            nodes = min(max(nodes, math.ceil(gpus / node_gpus)), num_nodes)
        prepared.append((job.idx, job.submit_s, gpus, nodes, job.duration_s))
    return prepared


def run_fcfs_consolidated(jobs: list[Job], num_nodes: int, node_gpus: int, oversized: str = "cap") -> list[Placement]:
    """Schedule ``jobs`` strict FCFS with consolidated (gang) placement that honours ``nodes``.

    As in :func:`run_fcfs`, jobs start in submission order and never before the previous one
    (``start = max(submit, last_start)``). Admission is node-aware: a job starts at the earliest time
    :func:`place_consolidated` fits it on the free GPUs per node, waiting for running jobs to finish if
    needed. :func:`_prepare_consolidated` handles ``oversized``: ``"drop"`` skips a job larger than
    the cluster and ``"cap"`` shrinks it; both widen ``nodes`` for any other infeasible job.
    """
    prepared = _prepare_consolidated(jobs, num_nodes, node_gpus, oversized)
    if not prepared:
        return []

    order = sorted(prepared, key=lambda t: t[1])  # by submission time; stable => FIFO on ties
    free = [node_gpus] * num_nodes
    running: list[tuple[float, tuple[tuple[int, int], ...]]] = []  # min-heap of (end_s, assignment)
    last_start = order[0][1]
    placements: list[Placement] = []
    for index, submit, gpus, nodes, duration in order:
        start = max(submit, last_start)  # FCFS: never start before the previous job
        while running and running[0][0] <= start:  # release everything finished by `start`
            _, freed = heapq.heappop(running)
            for node_id, count in freed:
                free[node_id] += count
        assigned = place_consolidated(free, gpus, nodes, node_gpus)
        while assigned is None:  # advance time until the consolidated placement fits
            if not running:  # pragma: no cover - _prepare_consolidated guarantees feasibility
                raise RuntimeError(f"consolidated placement infeasible for job {index}")
            end_s, freed = heapq.heappop(running)
            start = max(start, end_s)
            for node_id, count in freed:
                free[node_id] += count
            assigned = place_consolidated(free, gpus, nodes, node_gpus)
        for node_id, count in assigned:
            free[node_id] -= count
        nodes_assignment = tuple(assigned)
        heapq.heappush(running, (start + duration, nodes_assignment))
        placements.append(Placement(index, start, gpus, nodes_assignment))
        last_start = start
    return placements


def run_backfill_consolidated(
    jobs: list[Job], node_gpus: int, num_nodes: int, oversized: str = "cap"
) -> list[Placement]:
    """Schedule ``jobs`` FIFO with aggressive backfill and consolidated (gang) placement.

    Same event loop as :func:`run_backfill`: at each event time any queued job that fits starts, so a
    small later job can pass a blocked larger one. Placement uses :func:`place_consolidated`, so a job
    lands on exactly ``nodes`` distinct nodes. :func:`_prepare_consolidated` handles ``oversized``:
    ``"drop"`` skips a job larger than the cluster and ``"cap"`` shrinks it; both widen ``nodes`` for
    any other infeasible job.
    """
    prepared = _prepare_consolidated(jobs, num_nodes, node_gpus, oversized)
    if not prepared:
        return []

    arrivals = sorted(prepared, key=lambda t: t[1])  # by submission time; stable => FIFO on ties
    n = len(arrivals)
    pending: list[tuple[int, float, int, int, float]] = []
    running: list[tuple[float, tuple[tuple[int, int], ...]]] = []  # min-heap of (end_s, assignment)
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
        for queue_pos, (index, _submit, gpus, nodes, duration) in enumerate(pending):
            assigned = place_consolidated(free, gpus, nodes, node_gpus)
            if assigned is None:
                continue
            for node_id, count in assigned:
                free[node_id] -= count
            nodes_assignment = tuple(assigned)
            heapq.heappush(running, (time + duration, nodes_assignment))
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
