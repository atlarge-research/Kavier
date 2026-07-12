"""Cluster simulation: a fixed-size GPU cluster running jobs of known duration under a simple
scheduling policy (FCFS / backfill), reporting per-job queue/run metrics and per-cluster
makespan/utilisation.

The scheduling kernels live in ``core/`` (stdlib-only), and the public ``schedule`` verb plus its
result types live in ``facade.py`` and are re-exported here **lazily** (PEP 562 ``__getattr__``), so
a bare ``import kavier.sdk.cluster`` stays import-light and does not pull ``pandas``.
``kavier.cluster`` is a convenience alias for this package.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for type-checkers only — never imported at runtime (keeps import-time light)
    from kavier.sdk.cluster.facade import (
        ClusterMetrics as ClusterMetrics,
    )
    from kavier.sdk.cluster.facade import (
        ClusterSimResult as ClusterSimResult,
    )
    from kavier.sdk.cluster.facade import (
        JobRecord as JobRecord,
    )
    from kavier.sdk.cluster.facade import (
        schedule as schedule,
    )

_FACADE_EXPORTS = frozenset(
    {
        "schedule",
        "ClusterSimResult",
        "ClusterMetrics",
        "JobRecord",
    }
)


def __getattr__(name: str) -> Any:
    if name in _FACADE_EXPORTS:
        value = getattr(importlib.import_module(f"{__name__}.facade"), name)
        globals()[name] = value  # cache so subsequent access skips __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
