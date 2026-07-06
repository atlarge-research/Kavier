"""Kavier: predict performance, sustainability, and efficiency of LLM ecosystems.

Public API (lazily loaded, so ``import kavier`` stays cheap and stdlib-light):

* ``kavier.inference`` / ``kavier.training`` — the batch-predictor verbs
  (``performance`` / ``energy`` / ``efficiency`` / ``carbon``). Each takes a workload batch
  (a pandas DataFrame, a ``list[dict]``, or a single ``dict``) and returns the input rows plus the
  predicted columns.
* ``GPU_SPEC_LIBRARY`` / ``LLM_SPEC_LIBRARY`` — the static GPU/LLM spec catalogues.
* ``simulate_full_training`` / ``simulate_training_step`` — the low-level training engine.

The simulation engines live under ``kavier.sdk`` (``kavier.sdk.inference``, ``kavier.sdk.training``,
``kavier.sdk.energy``, ``kavier.sdk.co2``, ``kavier.sdk.io``, ``kavier.sdk.library``); the two verb
packages above are thin facades over them.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for type-checkers only — never imported at runtime (keeps import-time light)
    from kavier.sdk import inference as inference
    from kavier.sdk import training as training
    from kavier.sdk.library import GPU_SPEC_LIBRARY as GPU_SPEC_LIBRARY
    from kavier.sdk.library import LLM_SPEC_LIBRARY as LLM_SPEC_LIBRARY
    from kavier.sdk.training.core.engine import simulate_full_training as simulate_full_training
    from kavier.sdk.training.core.engine import simulate_training_step as simulate_training_step

__all__ = [
    "inference",
    "training",
    "GPU_SPEC_LIBRARY",
    "LLM_SPEC_LIBRARY",
    "simulate_full_training",
    "simulate_training_step",
]

# Lazy attribute access (PEP 562). Importing ``kavier`` must NOT pull pandas/numpy or any engine: this
# keeps ``import kavier`` cheap AND preserves the stdlib-only import contract of
# ``kavier.sdk.training.calibration`` (a bare ``import kavier.sdk.training.calibration`` executes this
# module first — see kavier/sdk/__init__.py). Heavy names resolve only on first access.
#
# ``kavier.inference`` / ``kavier.training`` are convenience aliases for the sdk verb packages: the
# functionality (engine + the performance/energy/efficiency/carbon verbs) lives under ``kavier.sdk.*``.
_LAZY_ALIASES = {
    "inference": "kavier.sdk.inference",
    "training": "kavier.sdk.training",
}
_LAZY_ATTRS = {
    "GPU_SPEC_LIBRARY": "kavier.sdk.library",
    "LLM_SPEC_LIBRARY": "kavier.sdk.library",
    "simulate_full_training": "kavier.sdk.training.core.engine",
    "simulate_training_step": "kavier.sdk.training.core.engine",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ALIASES.get(name)
    if target is not None:
        module = importlib.import_module(target)
        globals()[name] = module
        return module
    module_name = _LAZY_ATTRS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, "__version__"])


# Version from installed dist metadata; pyproject's static ``version`` is the single source of truth.
try:
    __version__ = version("kavier")
except PackageNotFoundError:  # editable/source tree without dist metadata
    __version__ = "0.0.0+unknown"
