"""Cluster simulation: a fixed-size GPU cluster running jobs of known duration under a simple
scheduling policy (FCFS / backfill), reporting per-job and per-cluster metrics.

Exports are re-exported lazily (PEP 562 ``__getattr__``) so a bare ``import kavier.sdk.cluster``
stays light and pulls in neither ``pandas`` nor ``matplotlib``. ``kavier.cluster`` aliases this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kavier._lazy import lazy_getattr

if TYPE_CHECKING:  # type-checkers only; never imported at runtime
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
        NodeRecord as NodeRecord,
    )
    from kavier.sdk.cluster.facade import (
        schedule as schedule,
    )
    from kavier.sdk.cluster.plot import (
        plot_timeline as plot_timeline,
    )

# Which submodule each export lazily comes from. ``plot`` imports matplotlib only when called.
_LAZY_EXPORTS = {
    "schedule": "facade",
    "ClusterSimResult": "facade",
    "ClusterMetrics": "facade",
    "JobRecord": "facade",
    "NodeRecord": "facade",
    "plot_timeline": "plot",
}

__getattr__ = lazy_getattr(globals(), attrs=_LAZY_EXPORTS)
