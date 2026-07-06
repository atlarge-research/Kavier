"""Training functionality: the analytical training engine + calibration plus the public verbs.

The first-principles step/full-run engine lives in ``core/`` (driven by the ``kavier training`` CLI),
wrapped by the ``calibration/`` correction layer. The batch predictors (``performance`` / ``energy`` /
``efficiency`` / ``carbon``) and the run/carbon helpers live in ``facade.py`` and are re-exported here
**lazily**. Keeping this ``__init__`` import-light is load-bearing: a bare
``import kavier.sdk.training.calibration`` executes this module, and the calibration accessor must stay
stdlib-only (no scipy/sklearn/numpy/pandas). ``kavier.training`` is a convenience alias for this package.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for type-checkers only — never imported at runtime (keeps import-time light)
    from kavier.sdk.training.facade import (
        DEFAULT_INTENSITY_G_KWH as DEFAULT_INTENSITY_G_KWH,
    )
    from kavier.sdk.training.facade import (
        DEFAULT_NUM_NODES as DEFAULT_NUM_NODES,
    )
    from kavier.sdk.training.facade import (
        carbon as carbon,
    )
    from kavier.sdk.training.facade import (
        efficiency as efficiency,
    )
    from kavier.sdk.training.facade import (
        energy as energy,
    )
    from kavier.sdk.training.facade import (
        performance as performance,
    )
    from kavier.sdk.training.facade import (
        run_carbon_from_training as run_carbon_from_training,
    )
    from kavier.sdk.training.facade import (
        run_training as run_training,
    )

# Public names re-exported (lazily) from facade.py. Listed explicitly so real submodules
# (facade / cli / core / calibration) are left to the normal import machinery — never delegated,
# and so this __init__ never imports the facade's heavy deps at module load (calibration contract).
_FACADE_EXPORTS = frozenset(
    {
        "performance",
        "energy",
        "efficiency",
        "carbon",
        "run_training",
        "run_carbon_from_training",
        "DEFAULT_INTENSITY_G_KWH",
        "DEFAULT_NUM_NODES",
    }
)


def __getattr__(name: str) -> Any:
    if name in _FACADE_EXPORTS:
        value = getattr(importlib.import_module(f"{__name__}.facade"), name)
        globals()[name] = value  # cache so subsequent access skips __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
