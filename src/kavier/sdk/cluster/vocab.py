"""Scheduling ``Policy`` and oversized-job ``Oversized`` enums for the cluster simulator.

``StrEnum`` members are ``str``, so ``Policy.CONSOLIDATED_FCFS == "consolidated-fcfs"`` and
``schedule(policy=...)`` accepts plain strings. The enums sit in a stdlib-only module to avoid an import
cycle: ``core/engine.py`` needs ``Oversized`` and ``facade.py`` imports the engine.
"""

from __future__ import annotations

from enum import StrEnum


class Policy(StrEnum):
    """Scheduling policy for :func:`kavier.sdk.cluster.schedule`.

    ``distributed-*`` tight-packs jobs and ignores each job's ``nodes`` request. ``consolidated-*``
    gang-places a job on ``nodes`` distinct nodes, one replica per node. ``*-fcfs`` is strict
    head-of-line first-come-first-served; ``*-backfill`` is FIFO with aggressive backfill.
    """

    DISTRIBUTED_FCFS = "distributed-fcfs"
    DISTRIBUTED_BACKFILL = "distributed-backfill"
    CONSOLIDATED_FCFS = "consolidated-fcfs"
    CONSOLIDATED_BACKFILL = "consolidated-backfill"


class Oversized(StrEnum):
    """How to treat a job that requests more GPUs than the cluster has: clamp it or skip it."""

    CAP = "cap"
    DROP = "drop"
