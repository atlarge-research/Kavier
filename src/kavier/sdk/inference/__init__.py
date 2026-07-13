"""Inference functionality: the per-request simulator engine plus the public batch-predictor verbs.

The simulator core lives in ``core/`` and ``stages/``. The batch predictors (``performance`` /
``energy`` / ``efficiency`` / ``carbon``) live in ``facade.py`` and are re-exported here lazily, so
importing this package stays cheap until a verb is first used. ``kavier.inference`` aliases this package.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type-checkers only; never imported at runtime
    from kavier.sdk.inference.facade import (
        DEFAULT_GPU_HOUR_PRICE as DEFAULT_GPU_HOUR_PRICE,
    )
    from kavier.sdk.inference.facade import (
        DEFAULT_INTENSITY_G_KWH as DEFAULT_INTENSITY_G_KWH,
    )
    from kavier.sdk.inference.facade import (
        DEFAULT_KV_CACHE as DEFAULT_KV_CACHE,
    )
    from kavier.sdk.inference.facade import (
        DEFAULT_PREFIX_MIN_TOKENS as DEFAULT_PREFIX_MIN_TOKENS,
    )
    from kavier.sdk.inference.facade import (
        DEFAULT_PREFIX_POLICY as DEFAULT_PREFIX_POLICY,
    )
    from kavier.sdk.inference.facade import (
        carbon as carbon,
    )
    from kavier.sdk.inference.facade import (
        efficiency as efficiency,
    )
    from kavier.sdk.inference.facade import (
        energy as energy,
    )
    from kavier.sdk.inference.facade import (
        energy_from_inference as energy_from_inference,
    )
    from kavier.sdk.inference.facade import (
        export_opendc as export_opendc,
    )
    from kavier.sdk.inference.facade import (
        performance as performance,
    )
    from kavier.sdk.inference.facade import (
        run_carbon_from_inference as run_carbon_from_inference,
    )
    from kavier.sdk.inference.facade import (
        run_inference as run_inference,
    )

# Public names re-exported (lazily) from facade.py. Listed explicitly so real submodules
# (facade / cli / core / stages) are left to the normal import machinery — never delegated.
_FACADE_EXPORTS = frozenset(
    {
        "performance",
        "energy",
        "efficiency",
        "carbon",
        "run_inference",
        "run_carbon_from_inference",
        "energy_from_inference",
        "export_opendc",
        "DEFAULT_INTENSITY_G_KWH",
        "DEFAULT_GPU_HOUR_PRICE",
        "DEFAULT_KV_CACHE",
        "DEFAULT_PREFIX_POLICY",
        "DEFAULT_PREFIX_MIN_TOKENS",
    }
)


def __getattr__(name: str) -> Any:
    if name in _FACADE_EXPORTS:
        value = getattr(importlib.import_module(f"{__name__}.facade"), name)
        globals()[name] = value  # cache so subsequent access skips __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
