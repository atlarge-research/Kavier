"""Simulate a fixed-size GPU cluster running jobs of known duration under FCFS or backfill scheduling.

Exports load lazily (PEP 562 ``__getattr__``), so ``import kavier.sdk.cluster`` imports neither pandas
nor matplotlib. ``kavier.cluster`` is an alias of this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kavier._lazy import lazy_getattr

if TYPE_CHECKING:
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

# Export name -> submodule; ``plot`` imports matplotlib inside plot_timeline.
_LAZY_EXPORTS = {
    "schedule": "facade",
    "ClusterSimResult": "facade",
    "ClusterMetrics": "facade",
    "JobRecord": "facade",
    "NodeRecord": "facade",
    "plot_timeline": "plot",
}

__getattr__ = lazy_getattr(globals(), attrs=_LAZY_EXPORTS)
