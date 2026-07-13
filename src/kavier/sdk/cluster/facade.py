"""Public ``kavier.sdk.cluster`` verb: ``schedule(jobs, ...) -> ClusterSimResult``.

Simulates a fixed-size GPU cluster running jobs of known duration under a scheduling policy (``"fcfs"``
or ``"backfill"``) and returns per-job metrics (wait, start/end, runtime, energy), per-cluster metrics
(makespan, utilisation, goodput, peaks), and a GPUs-in-use / queue-depth timeline.

The scheduling kernels live in :mod:`kavier.sdk.cluster.core.engine`; this facade adds input
normalisation, energy, and metrics. ``pandas`` is imported lazily (only for a DataFrame input) so a
bare import stays light.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kavier.sdk.cluster.core import engine
from kavier.sdk.cluster.core import metrics as _metrics
from kavier.sdk.units import SECONDS_PER_HOUR, WS_PER_KWH

_POLICIES = ("fcfs", "backfill")
_OVERSIZED = ("cap", "drop")


@dataclass(frozen=True)
class JobRecord:
    """Per-job simulation outcome. Canonical unit is seconds; ``*_h`` helpers give hours."""

    job_id: Any
    gpus: int  # GPUs actually placed (after any oversized cap)
    submit_s: float
    start_s: float
    end_s: float
    wait_s: float  # start - submit (time queued)
    runtime_s: float  # end - start (equals the job's duration)
    turnaround_s: float  # end - submit (wait + runtime)
    energy_kwh: float | None  # None when no per-GPU power is known
    nodes: tuple[tuple[int, int], ...]  # ((node_id, gpus_on_node), ...) the job was placed on

    @property
    def submit_h(self) -> float:
        return self.submit_s / SECONDS_PER_HOUR

    @property
    def start_h(self) -> float:
        return self.start_s / SECONDS_PER_HOUR

    @property
    def end_h(self) -> float:
        return self.end_s / SECONDS_PER_HOUR

    @property
    def wait_h(self) -> float:
        return self.wait_s / SECONDS_PER_HOUR

    @property
    def runtime_h(self) -> float:
        return self.runtime_s / SECONDS_PER_HOUR

    @property
    def turnaround_h(self) -> float:
        return self.turnaround_s / SECONDS_PER_HOUR


@dataclass(frozen=True)
class ClusterMetrics:
    """Per-cluster aggregate metrics over the scheduled jobs."""

    n_jobs: int
    capacity_gpus: int
    makespan_s: float
    avg_wait_s: float
    avg_run_s: float
    avg_turnaround_s: float
    utilization: float  # GPU·s used / (capacity × makespan)
    goodput_jobs_per_s: float
    total_energy_kwh: float | None
    peak_gpus: int
    peak_queue: int

    @property
    def makespan_h(self) -> float:
        return self.makespan_s / SECONDS_PER_HOUR

    @property
    def avg_wait_h(self) -> float:
        return self.avg_wait_s / SECONDS_PER_HOUR

    @property
    def avg_run_h(self) -> float:
        return self.avg_run_s / SECONDS_PER_HOUR

    @property
    def avg_turnaround_h(self) -> float:
        return self.avg_turnaround_s / SECONDS_PER_HOUR


@dataclass(frozen=True)
class NodeRecord:
    """Per-node aggregate over the scheduled jobs. Canonical time unit is seconds."""

    node_id: int
    gpus: int  # GPUs on this node (= node_gpus)
    jobs_hosted: int  # jobs that placed >=1 GPU on this node
    busy_gpu_s: float  # Σ (gpus-on-node × runtime_s) over hosted jobs
    utilization: float  # busy_gpu_s / (gpus × makespan_s)
    peak_gpus_used: int  # max concurrent GPUs in use on this node
    idle_s: float  # wall-seconds with zero GPUs in use over the makespan window
    energy_kwh: float | None  # apportioned per-job energy; None when no hosted job had power


@dataclass(frozen=True)
class Timeline:
    """Aligned step-series over one shared time axis (seconds); ``*_h`` gives hours."""

    times_s: list[float]
    gpus_in_use: list[float]
    queue_depth: list[float]

    @property
    def times_h(self) -> list[float]:
        return [t / SECONDS_PER_HOUR for t in self.times_s]


@dataclass(frozen=True)
class ClusterSimResult:
    """Result of :func:`schedule`: scheduled jobs, cluster metrics, timeline, node metrics, drops."""

    policy: str
    jobs: list[JobRecord]
    cluster: ClusterMetrics
    timeline: Timeline
    dropped: list[Any]
    nodes: list[NodeRecord]


def _normalise(jobs: Any) -> list[dict[str, Any]]:
    """Coerce ``list[dict] | list[tuple] | pandas.DataFrame`` into canonical job dicts.

    Canonical keys: ``submit_s``, ``gpus``, ``duration_s`` (required); ``nodes`` (default 1),
    ``power_w_per_gpu`` (default None), ``job_id`` (default the row index).
    """
    if hasattr(jobs, "to_dict") and hasattr(jobs, "columns"):  # duck-typed pandas DataFrame
        rows: list[Mapping[str, Any]] = jobs.to_dict(orient="records")
    else:
        rows = list(jobs)

    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            submit_s = row.get("submit_s", row.get("submit"))
            gpus = row.get("gpus")
            duration_s = row.get("duration_s", row.get("duration"))
            nodes = row.get("nodes", 1)
            power = row.get("power_w_per_gpu")
            job_id = row.get("job_id", index)
        elif isinstance(row, (Sequence, tuple)) and not isinstance(row, (str, bytes)):
            submit_s, gpus, duration_s = row[0], row[1], row[2]
            nodes = row[3] if len(row) > 3 else 1
            power = None
            job_id = index
        else:
            raise TypeError(f"job {index} must be a mapping or a (submit_s, gpus, duration_s[, nodes]) tuple")
        if submit_s is None or gpus is None or duration_s is None:
            raise ValueError(f"job {index} needs submit_s, gpus and duration_s")
        submit_f = float(submit_s)
        duration_f = float(duration_s)
        if math.isnan(submit_f) or math.isnan(duration_f):
            raise ValueError(f"job {index}: submit_s and duration_s must be finite numbers")
        # A blank/NaN power means "unknown" (energy stays None), not a NaN poisoning the total.
        power_f = None if power is None else float(power)
        if power_f is not None and math.isnan(power_f):
            power_f = None
        out.append(
            {
                "index": index,
                "job_id": job_id,
                "submit_s": submit_f,
                "gpus": int(gpus),
                "duration_s": duration_f,
                "nodes": int(nodes) if nodes else 1,
                "power_w_per_gpu": power_f,
            }
        )
    return out


def schedule(
    jobs: Any,
    *,
    policy: str = "fcfs",
    num_nodes: int | None = None,
    node_gpus: int | None = None,
    oversized: str = "cap",
    default_watts_per_gpu: float | None = None,
) -> ClusterSimResult:
    """Simulate ``jobs`` on a homogeneous ``num_nodes × node_gpus`` datacenter and return per-job,
    per-cluster, and per-node metrics.

    ``jobs`` is a ``list[dict]`` / ``list[tuple]`` / ``pandas.DataFrame`` of
    ``submit_s, gpus, duration_s[, power_w_per_gpu, job_id]`` (the ``nodes`` column is ignored;
    placement is automatic tight-pack). ``policy="fcfs"`` is strict FCFS timing; ``policy="backfill"``
    is FIFO+backfill. ``oversized`` is ``"cap"`` (clamp a too-big job to the cluster) or ``"drop"``
    (skip it). Energy per job is ``(power_w_per_gpu or default_watts_per_gpu) × gpus × runtime_s /
    3.6e6`` kWh (``None`` if no power).
    """
    if policy not in _POLICIES:
        raise ValueError(f"policy must be one of {_POLICIES}, got {policy!r}")
    if oversized not in _OVERSIZED:
        raise ValueError(f"oversized must be one of {_OVERSIZED}, got {oversized!r}")
    if num_nodes is None or node_gpus is None:
        raise ValueError("cluster needs num_nodes and node_gpus (e.g. num_nodes=4, node_gpus=8)")
    num_nodes = int(num_nodes)
    node_gpus = int(node_gpus)
    if num_nodes < 1 or node_gpus < 1:
        raise ValueError(f"num_nodes and node_gpus must be >= 1, got {num_nodes} and {node_gpus}")
    capacity = num_nodes * node_gpus

    norm = _normalise(jobs)
    ejobs = [engine.Job(j["index"], j["submit_s"], j["gpus"], j["duration_s"], j["nodes"]) for j in norm]

    if policy == "fcfs":
        placements = engine.run_fcfs(ejobs, num_nodes, node_gpus, oversized)
    else:
        placements = engine.run_backfill(ejobs, node_gpus, num_nodes, oversized)

    placed = {p.idx: p for p in placements}
    records: list[JobRecord] = []
    for job in norm:
        placement = placed.get(job["index"])
        if placement is None:
            continue
        start_s = placement.start_s
        gpus = placement.gpus
        runtime_s = job["duration_s"]
        end_s = start_s + runtime_s
        power = job["power_w_per_gpu"] if job["power_w_per_gpu"] is not None else default_watts_per_gpu
        energy_kwh = None if power is None else power * gpus * runtime_s / WS_PER_KWH
        records.append(
            JobRecord(
                job_id=job["job_id"],
                gpus=gpus,
                submit_s=job["submit_s"],
                start_s=start_s,
                end_s=end_s,
                wait_s=start_s - job["submit_s"],
                runtime_s=runtime_s,
                turnaround_s=end_s - job["submit_s"],
                energy_kwh=energy_kwh,
                nodes=placement.nodes,
            )
        )

    dropped = [job["job_id"] for job in norm if job["index"] not in placed]
    cluster, timeline = _summarise(records, capacity)
    nodes = _node_records(records, num_nodes, node_gpus)
    return ClusterSimResult(
        policy=policy, jobs=records, cluster=cluster, timeline=timeline, dropped=dropped, nodes=nodes
    )


def _node_records(records: list[JobRecord], num_nodes: int, node_gpus: int) -> list[NodeRecord]:
    """Per-node aggregates over the makespan window ``[min start, max end]``.

    Node energy is the per-job energy apportioned by GPU fraction (``energy_kwh × gpus_on_node /
    gpus``), so per-node energies sum to the cluster total and a job with unknown power contributes
    nothing (never a poisoned NaN).
    """
    if not records:
        return [
            NodeRecord(
                node_id=n,
                gpus=node_gpus,
                jobs_hosted=0,
                busy_gpu_s=0.0,
                utilization=0.0,
                peak_gpus_used=0,
                idle_s=0.0,
                energy_kwh=None,
            )
            for n in range(num_nodes)
        ]
    t0 = min(r.start_s for r in records)
    t_end = max(r.end_s for r in records)
    makespan_s = t_end - t0
    intervals: dict[int, list[tuple[float, float, int]]] = {n: [] for n in range(num_nodes)}
    jobs_hosted = {n: 0 for n in range(num_nodes)}
    busy_gpu_s = {n: 0.0 for n in range(num_nodes)}
    energy = {n: 0.0 for n in range(num_nodes)}
    energy_known = {n: False for n in range(num_nodes)}
    for r in records:
        for node_id, count in r.nodes:
            intervals[node_id].append((r.start_s, r.end_s, count))
            jobs_hosted[node_id] += 1
            busy_gpu_s[node_id] += count * r.runtime_s
            if r.energy_kwh is not None and r.gpus > 0:
                energy[node_id] += r.energy_kwh * count / r.gpus
                energy_known[node_id] = True
    out: list[NodeRecord] = []
    for node_id in range(num_nodes):
        peak, idle = _metrics.node_activity(intervals[node_id], t0, t_end)
        util = busy_gpu_s[node_id] / (node_gpus * makespan_s) if node_gpus > 0 and makespan_s > 0 else 0.0
        out.append(
            NodeRecord(
                node_id=node_id,
                gpus=node_gpus,
                jobs_hosted=jobs_hosted[node_id],
                busy_gpu_s=busy_gpu_s[node_id],
                utilization=util,
                peak_gpus_used=peak,
                idle_s=idle,
                energy_kwh=energy[node_id] if energy_known[node_id] else None,
            )
        )
    return out


def _summarise(records: list[JobRecord], capacity_gpus: int) -> tuple[ClusterMetrics, Timeline]:
    """Per-cluster metrics + the aligned GPUs-in-use / queue-depth timeline over the scheduled jobs."""
    if not records:
        empty = ClusterMetrics(
            n_jobs=0,
            capacity_gpus=capacity_gpus,
            makespan_s=0.0,
            avg_wait_s=0.0,
            avg_run_s=0.0,
            avg_turnaround_s=0.0,
            utilization=0.0,
            goodput_jobs_per_s=0.0,
            total_energy_kwh=None,
            peak_gpus=0,
            peak_queue=0,
        )
        return empty, Timeline(times_s=[], gpus_in_use=[], queue_depth=[])

    n = len(records)
    makespan_s = max(r.end_s for r in records) - min(r.start_s for r in records)
    avg_wait_s = sum(r.wait_s for r in records) / n
    avg_run_s = sum(r.runtime_s for r in records) / n
    avg_turnaround_s = sum(r.turnaround_s for r in records) / n
    gpu_seconds = sum(r.gpus * r.runtime_s for r in records)  # area under the GPUs-in-use curve
    utilization = gpu_seconds / (capacity_gpus * makespan_s) if capacity_gpus > 0 and makespan_s > 0 else 0.0
    goodput = n / makespan_s if makespan_s > 0 else 0.0
    energies = [r.energy_kwh for r in records if r.energy_kwh is not None]
    total_energy = sum(energies) if energies else None

    # Timeline events, shifted so the axis starts at the first arrival.
    t0 = min(r.submit_s for r in records)
    gpu_events: list[tuple[float, float]] = []
    queue_events: list[tuple[float, float]] = []
    for r in records:
        gpu_events.append((r.start_s - t0, float(r.gpus)))  # claim GPUs at start
        gpu_events.append((r.end_s - t0, -float(r.gpus)))  # release them at end
        queue_events.append((r.submit_s - t0, 1.0))  # enter the queue at submit
        queue_events.append((r.start_s - t0, -1.0))  # leave it at start
    t_end = max(r.end_s for r in records) - t0
    times, gpus_series, queue_series = _metrics.build_timeline(gpu_events, queue_events, t_end)
    peak_gpus = int(max(gpus_series)) if gpus_series else 0
    peak_queue = int(max(queue_series)) if queue_series else 0

    metrics = ClusterMetrics(
        n_jobs=n,
        capacity_gpus=capacity_gpus,
        makespan_s=makespan_s,
        avg_wait_s=avg_wait_s,
        avg_run_s=avg_run_s,
        avg_turnaround_s=avg_turnaround_s,
        utilization=utilization,
        goodput_jobs_per_s=goodput,
        total_energy_kwh=total_energy,
        peak_gpus=peak_gpus,
        peak_queue=peak_queue,
    )
    return metrics, Timeline(times_s=times, gpus_in_use=gpus_series, queue_depth=queue_series)
