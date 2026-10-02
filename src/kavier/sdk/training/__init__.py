"""Analytical training engine, its calibration layer, and the batch predictors.

``core/`` holds the step and full-run engine, corrected by the fitted tables in ``calibration/``.
``facade.py`` defines ``performance``, ``energy``, ``efficiency`` and ``carbon``; they are re-exported
here lazily. This ``__init__`` stays import-light because ``import kavier.sdk.training.calibration``
runs it, and the calibration accessor has to stay stdlib-only (no scipy, sklearn, numpy or pandas).
``kavier.training`` aliases this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kavier._lazy import lazy_getattr

if TYPE_CHECKING:
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

# Facade names resolved lazily. Submodules (facade, cli, core, calibration) are not listed, so they load
# through the normal import machinery and this module never imports the facade's dependencies.
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

__getattr__ = lazy_getattr(globals(), attrs={name: "facade" for name in _FACADE_EXPORTS})
