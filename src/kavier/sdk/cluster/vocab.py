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
    """How to treat a job requesting more GPUs than the whole cluster.

    ``"cap"`` clamps the job's GPU count to the cluster total; ``"drop"`` silently skips it;
    ``"strict"`` raises a :class:`ValueError` before simulation begins if any job exceeds capacity.
    """

    CAP = "cap"
    DROP = "drop"
    STRICT = "strict"


class PlacementStrategy(StrEnum):
    """Node-selection strategy for consolidated placement (:func:`place_consolidated`).

    ``"pack"`` prefers the least-free node first (bin-packing / tightest-fit), filling nodes before
    moving on to fresh ones — keeps whole nodes open for large jobs. ``"spread"`` prefers the
    most-free node first, mirroring the Kubernetes ``LeastAllocated`` scorer: jobs are distributed
    evenly across nodes rather than consolidated onto as few as possible.

    Only affects the ``consolidated-*`` scheduling policies; the ``distributed-*`` policies use
    :func:`place` (tight-pack only) and are unaffected.
    """

    PACK = "pack"
    SPREAD = "spread"
