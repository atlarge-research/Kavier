"""Discrete-event cluster schedulers: strict FCFS (flat pool) and node-aware backfill.

Stdlib only (``heapq`` + ``math``) — this module is the import-light simulation kernel and must
never pull ``pandas``/``numpy``/the spec library. It is a faithful port of the two frozen thesis
schedulers (reproducibility-capsule ``gen_exp2.py::schedule`` and ``gen_exp4.py::schedule_backfill``);
parity with those originals is guarded by ``tests/test_cluster/test_schedule_parity.py``.

Both kernels take a list of :class:`Job` tuples and return a list of :class:`Placement` tuples
(one per SCHEDULED job — a job dropped for being oversized has no placement). All times are in
**seconds**; a job's runtime equals its ``duration_s`` (a started job runs to completion).
"""

from __future__ import annotations

import heapq
import math
from typing import NamedTuple


class Job(NamedTuple):
    """A schedulable job. ``nodes`` is honoured only by the node-aware backfill kernel."""

    idx: int  # position in the caller's job list (NOT named ``index`` — that shadows tuple.index)
    submit_s: float
    gpus: int
    duration_s: float
    nodes: int


class Placement(NamedTuple):
    """The scheduler's decision for one job: when it starts and how many GPUs it was placed on."""

    idx: int
    start_s: float
    gpus: int


def run_fcfs(jobs: list[Job], capacity_gpus: int, oversized: str = "cap") -> list[Placement]:
    """Strict First-Come-First-Served on a flat pool of ``capacity_gpus`` GPUs (no backfill).

    Port of ``gen_exp2.py::schedule``: jobs run in submission order and a job can never start
    before the previous job in that order (``start = max(submit, last_start)``), so an oversized or
    starved head-of-line job stalls everything behind it. ``oversized="cap"`` clamps a job that wants
    more than the whole cluster to ``capacity_gpus`` (the frozen behaviour); ``oversized="drop"``
    skips it entirely (the coastline-UI behaviour — a job that can never fit would block FIFO forever).
    """
    active: list[tuple[int, float, int, float]] = []
    for job in jobs:
        gpus = job.gpus
        if gpus > capacity_gpus:
            if oversized == "drop":
                continue
            gpus = capacity_gpus  # cap
        active.append((job.idx, job.submit_s, gpus, job.duration_s))
    if not active:
        return []

    order = sorted(active, key=lambda t: t[1])  # by submission time; stable => FIFO on ties
    running: list[tuple[float, int]] = []  # min-heap of (end_s, gpus)
    free = capacity_gpus
    last_start = order[0][1]
    placements: list[Placement] = []
    for index, submit, gpus, duration in order:
        start = max(submit, last_start)  # FCFS: never start before the previous job
        while free < gpus:  # wait for running jobs to free enough GPUs
            end_s, freed = heapq.heappop(running)
            start = max(start, end_s)
            free += freed
        free -= gpus
        heapq.heappush(running, (start + duration, gpus))
        placements.append(Placement(index, start, gpus))
        last_start = start
    return placements


def _placement(free: list[int], gpus_per_node: int, num_nodes: int) -> list[int] | None:
    """Tightest-fit node ids for a job needing ``gpus_per_node`` GPUs on ``num_nodes`` distinct
    nodes, or ``None`` if it does not fit the currently free GPUs (keeps big jobs' whole nodes free)."""
    fitting = sorted((f, n) for n, f in enumerate(free) if f >= gpus_per_node)
    if len(fitting) < num_nodes:
        return None
    return [n for _, n in fitting[:num_nodes]]


def run_backfill(jobs: list[Job], node_gpus: int, num_nodes: int, oversized: str = "cap") -> list[Placement]:
    """Best-effort FIFO with aggressive backfill on a ``num_nodes × node_gpus`` cluster.

    Port of ``gen_exp4.py::schedule_backfill`` (and coastline ``plot.py::schedule_backfill``): jobs
    are considered in submission order every tick, and any queued job whose whole-node placement fits
    starts now — so a small later job backfills past an earlier job that does not fit (no head-of-line
    reservation). A job is placed on ``nodes`` whole nodes at ``per_node = min(ceil(gpus/nodes),
    node_gpus)`` each, so an oversized request is capped by the node geometry (``cap`` semantics);
    ``oversized="drop"`` skips a job that wants more than the whole cluster.
    """
    total = node_gpus * num_nodes
    unpacked: list[tuple[int, float, int, int, float]] = []
    for job in jobs:
        if oversized == "drop" and job.gpus > total:
            continue
        nodes = max(1, min(int(job.nodes), num_nodes))
        per_node = min(math.ceil(job.gpus / nodes), node_gpus)
        unpacked.append((job.idx, job.submit_s, per_node, nodes, job.duration_s))
    if not unpacked:
        return []

    arrivals = sorted(unpacked, key=lambda t: t[1])  # by submission time; stable => FIFO on ties
    n = len(arrivals)
    pending: list[tuple[int, float, int, int, float]] = []
    running: list[tuple[float, int, tuple[int, ...]]] = []  # min-heap of (end_s, per_node, node_ids)
    free = [node_gpus] * num_nodes
    next_arrival = 0
    done: dict[int, Placement] = {}

    time = arrivals[0][1]
    while len(done) < n:
        while next_arrival < n and arrivals[next_arrival][1] <= time:
            pending.append(arrivals[next_arrival])
            next_arrival += 1
        while running and running[0][0] <= time:
            _, freed, node_ids = heapq.heappop(running)
            for nd in node_ids:
                free[nd] += freed
        admitted: list[int] = []
        for queue_pos, (index, _submit, per_node, nodes, duration) in enumerate(pending):
            place_nodes = _placement(free, per_node, nodes)
            if place_nodes is None:
                continue
            for nd in place_nodes:
                free[nd] -= per_node
            heapq.heappush(running, (time + duration, per_node, tuple(place_nodes)))
            done[index] = Placement(index, time, per_node * len(place_nodes))
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
