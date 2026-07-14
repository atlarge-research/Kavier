"""Cluster-scheduling vocabularies: the scheduling ``Policy`` and the oversized-job ``Oversized`` mode.

``StrEnum`` members are ``str``, so ``Policy.CONSOLIDATED_FCFS == "consolidated-fcfs"`` and the public
``schedule(policy=...)`` boundary keeps accepting plain strings unchanged. These live in their own
stdlib-only module (not in ``facade.py``) because the scheduling kernel in ``core/engine.py`` needs
``Oversized`` while ``facade.py`` imports the engine — defining them in the facade would create an
import cycle.
"""

from __future__ import annotations

from enum import StrEnum


class Policy(StrEnum):
    """Scheduling discipline for :func:`kavier.sdk.cluster.schedule`.

    ``distributed-*`` spread jobs (tight-pack, ignoring each job's ``nodes`` request); ``consolidated-*``
    gang-place, honouring ``nodes`` (one replica per distinct co-located node). ``*-fcfs`` is strict
    head-of-line first-come-first-served; ``*-backfill`` is FIFO with aggressive backfill.
    """

    DISTRIBUTED_FCFS = "distributed-fcfs"
    DISTRIBUTED_BACKFILL = "distributed-backfill"
    CONSOLIDATED_FCFS = "consolidated-fcfs"
    CONSOLIDATED_BACKFILL = "consolidated-backfill"


class Oversized(StrEnum):
    """How to treat a job requesting more GPUs than the whole cluster: clamp it or skip it."""

    CAP = "cap"
    DROP = "drop"
