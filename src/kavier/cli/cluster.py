"""``kavier cluster`` subcommand: simulate a fixed GPU cluster running jobs of known duration.

Reads a jobs CSV (``submit_s,gpus,duration_s[,nodes,power_w_per_gpu,job_id]``), runs the FIFO/backfill
cluster simulator, prints the per-cluster summary as JSON to stdout, and optionally writes the per-job
schedule to a CSV. All modelling lives in :mod:`kavier.sdk.cluster`; this module is CLI glue only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kavier.cli._shared import FriendlyParser, apply_config
from kavier.sdk.cluster import schedule
from kavier.sdk.cluster.facade import ClusterSimResult
from kavier.sdk.cluster.vocab import Oversized, Policy

_EXAMPLE_CMD = "kavier cluster --jobs jobs.csv --policy fcfs --num-nodes 4 --node-gpus 8"

_PER_JOB_FIELDS = (
    "job_id",
    "gpus",
    "submit_s",
    "start_s",
    "end_s",
    "wait_s",
    "runtime_s",
    "turnaround_s",
    "energy_kwh",
    "node:gpus",
    "placement",
)

_PER_NODE_FIELDS = (
    "node_id",
    "gpus",
    "jobs_hosted",
    "busy_gpu_s",
    "utilization",
    "peak_gpus_used",
    "idle_s",
    "energy_kwh",
)


def add_cluster_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``kavier cluster`` flags to ``parser``."""
    parser.add_argument(
        "--jobs", required=True, help="Jobs CSV: submit_s,gpus,duration_s[,nodes,power_w_per_gpu,job_id]"
    )
    parser.add_argument(
        "--policy", choices=[p.value for p in Policy], default=Policy.FCFS, help="Scheduling policy (default: fcfs)"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes in the datacenter")
    parser.add_argument("--node-gpus", type=int, default=None, help="GPUs per node")
    parser.add_argument(
        "--oversized",
        choices=[o.value for o in Oversized],
        default=Oversized.CAP,
        help="Clamp (cap) or skip (drop) a job that wants more GPUs than the cluster (default: cap)",
    )
    parser.add_argument(
        "--watts-per-gpu", type=float, default=None, help="Fallback per-GPU power (W) for the energy estimate"
    )
    parser.add_argument("--out", default=None, help="Optional path to write the per-job schedule CSV")
    parser.add_argument("--out-nodes", default=None, help="Optional path to write the per-node CSV")
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional path to render the cluster-timeline figure (.pdf/.png; needs the [plot] extra)",
    )
    parser.add_argument("--config", default=None, help="Optional YAML config supplying flag defaults")
    return parser


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    """Parse the jobs CSV into canonical job dicts for :func:`kavier.sdk.cluster.schedule`."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"submit_s", "gpus", "duration_s"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"jobs CSV missing required column(s): {', '.join(sorted(missing))}")
        jobs: list[dict[str, Any]] = []
        for row in reader:
            job: dict[str, Any] = {
                "submit_s": float(row["submit_s"]),
                "gpus": int(row["gpus"]),
                "duration_s": float(row["duration_s"]),
            }
            if row.get("nodes"):
                job["nodes"] = int(row["nodes"])
            if row.get("power_w_per_gpu"):
                job["power_w_per_gpu"] = float(row["power_w_per_gpu"])
            if row.get("job_id"):
                job["job_id"] = row["job_id"]
            jobs.append(job)
    return jobs


def _summary(result: ClusterSimResult) -> dict[str, Any]:
    """Per-cluster summary for stdout (JSON-serialisable)."""
    cluster = result.cluster
    return {
        "policy": result.policy,
        "n_jobs": cluster.n_jobs,
        "capacity_gpus": cluster.capacity_gpus,
        "makespan_s": cluster.makespan_s,
        "makespan_h": cluster.makespan_h,
        "utilization": cluster.utilization,
        "avg_wait_s": cluster.avg_wait_s,
        "avg_run_s": cluster.avg_run_s,
        "avg_turnaround_s": cluster.avg_turnaround_s,
        "goodput_jobs_per_s": cluster.goodput_jobs_per_s,
        "total_energy_kwh": cluster.total_energy_kwh,
        "peak_gpus": cluster.peak_gpus,
        "peak_queue": cluster.peak_queue,
        "dropped": result.dropped,
    }


def _format_nodes(nodes: tuple[tuple[int, int], ...]) -> str:
    """Render a node assignment as ``"0:8;1:2"`` (node_id:gpus, semicolon-separated)."""
    return ";".join(f"{node_id}:{gpus}" for node_id, gpus in nodes)


def _describe_nodes(nodes: tuple[tuple[int, int], ...]) -> str:
    """Human-readable placement, e.g. ``"8 GPUs on node 1 + 1 GPU on node 2"``."""
    return " + ".join(f"{gpus} GPU{'s' if gpus != 1 else ''} on node {node_id}" for node_id, gpus in nodes)


_COMPUTED_JOB_FIELDS = ("node:gpus", "placement")


def _write_per_job(result: ClusterSimResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_PER_JOB_FIELDS))
        writer.writeheader()
        for job in result.jobs:
            row = {field: getattr(job, field) for field in _PER_JOB_FIELDS if field not in _COMPUTED_JOB_FIELDS}
            row["node:gpus"] = _format_nodes(job.nodes)
            row["placement"] = _describe_nodes(job.nodes)
            writer.writerow(row)


def _write_per_node(result: ClusterSimResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_PER_NODE_FIELDS))
        writer.writeheader()
        for node in result.nodes:
            writer.writerow({field: getattr(node, field) for field in _PER_NODE_FIELDS})


def main(argv: Sequence[str] | None = None) -> None:
    """Run the cluster simulator over a jobs CSV and print the summary (JSON) to stdout."""
    parser = add_cluster_args(FriendlyParser(prog="kavier cluster", example=_EXAMPLE_CMD))
    apply_config(parser, argv)
    args = parser.parse_args(argv)

    jobs_path = Path(args.jobs).expanduser()
    if not jobs_path.exists():
        parser.error(f"jobs file not found: {jobs_path}")

    try:
        jobs = _load_jobs(jobs_path)
        result = schedule(
            jobs,
            policy=args.policy,
            num_nodes=args.num_nodes,
            node_gpus=args.node_gpus,
            oversized=args.oversized,
            default_watts_per_gpu=args.watts_per_gpu,
        )
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))

    print(json.dumps(_summary(result), indent=2))
    if args.out:
        out_path = Path(args.out).expanduser()
        _write_per_job(result, out_path)
        print(f"Per-job schedule → {out_path}", file=sys.stderr)
    if args.out_nodes:
        out_nodes_path = Path(args.out_nodes).expanduser()
        _write_per_node(result, out_nodes_path)
        print(f"Per-node schedule → {out_nodes_path}", file=sys.stderr)
    if args.plot:
        from kavier.sdk.cluster import plot_timeline

        plot_path = Path(args.plot).expanduser()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plot_timeline(result, str(plot_path))
        except ImportError as exc:
            parser.error(str(exc))
        print(f"Cluster timeline → {plot_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
