"""Cluster simulation: a fixed-size GPU cluster running jobs of known duration under a simple
scheduling policy (FCFS / backfill), reporting per-job queue/run metrics and per-cluster
makespan/utilisation.

The scheduling kernels live in ``core/`` (stdlib-only), the public ``schedule`` verb plus its result
types live in ``facade.py``, and the optional matplotlib renderer ``plot_timeline`` lives in
``plot.py`` — all re-exported here **lazily** (PEP 562 ``__getattr__``), so a bare
``import kavier.sdk.cluster`` stays import-light and does not pull ``pandas`` or ``matplotlib``.
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
    from kavier.sdk.cluster.plot import (
        plot_timeline as plot_timeline,
    )

# name -> submodule it is (lazily) re-exported from. ``plot_timeline`` lives in ``plot`` (which imports
# matplotlib only when called), the rest in ``facade``.
_LAZY_EXPORTS = {
    "schedule": "facade",
    "ClusterSimResult": "facade",
    "ClusterMetrics": "facade",
    "JobRecord": "facade",
    "plot_timeline": "plot",
}


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is not None:
        value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
        globals()[name] = value  # cache so subsequent access skips __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
