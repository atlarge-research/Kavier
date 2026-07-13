"""Cluster-scheduling vocabularies: the scheduling ``Policy`` and the oversized-job ``Oversized`` mode.

``StrEnum`` members are ``str``, so ``Policy.FCFS == "fcfs"`` and the public ``schedule(policy=...)``
boundary keeps accepting plain strings unchanged. These live in their own stdlib-only module (not in
``facade.py``) because the scheduling kernel in ``core/engine.py`` needs ``Oversized`` while
``facade.py`` imports the engine — defining them in the facade would create an import cycle.
"""

from __future__ import annotations

from enum import StrEnum


class Policy(StrEnum):
    """Scheduling discipline for :func:`kavier.sdk.cluster.schedule`."""

    FCFS = "fcfs"
    BACKFILL = "backfill"


class Oversized(StrEnum):
    """How to treat a job requesting more GPUs than the whole cluster: clamp it or skip it."""

    CAP = "cap"
    DROP = "drop"
