"""Kavier: predict performance, sustainability, and efficiency of LLM ecosystems.

The public names (``inference``, ``training``, ``cluster``, the GPU and LLM spec libraries, and the
training engine functions) load on first access via PEP 562 ``__getattr__``. The engines live under
``kavier.sdk``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from kavier._lazy import lazy_getattr
from kavier.sdk.domain import Domain

if TYPE_CHECKING:
    from kavier.sdk import cluster as cluster
    from kavier.sdk import inference as inference
    from kavier.sdk import training as training
    from kavier.sdk.library import GPU_SPEC_LIBRARY as GPU_SPEC_LIBRARY
    from kavier.sdk.library import LLM_SPEC_LIBRARY as LLM_SPEC_LIBRARY
    from kavier.sdk.training.core.engine import simulate_full_training as simulate_full_training
    from kavier.sdk.training.core.engine import simulate_training_step as simulate_training_step

__all__ = [
    "cluster",
    "inference",
    "training",
    "GPU_SPEC_LIBRARY",
    "LLM_SPEC_LIBRARY",
    "simulate_full_training",
    "simulate_training_step",
]

# Keep this module free of pandas/numpy: ``import kavier.sdk.training.calibration`` runs it first
# and must stay stdlib-only. Targets are submodule paths relative to ``kavier``.
_LAZY_ALIASES = {
    "cluster": "sdk.cluster",
    Domain.INFERENCE: "sdk.inference",
    Domain.TRAINING: "sdk.training",
}
_LAZY_ATTRS = {
    "GPU_SPEC_LIBRARY": "sdk.library",
    "LLM_SPEC_LIBRARY": "sdk.library",
    "simulate_full_training": "sdk.training.core.engine",
    "simulate_training_step": "sdk.training.core.engine",
}

__getattr__ = lazy_getattr(globals(), modules=_LAZY_ALIASES, attrs=_LAZY_ATTRS)


def __dir__() -> list[str]:
    return sorted([*__all__, "__version__"])


# Read from the installed dist metadata, which comes from the static version in pyproject.toml.
try:
    __version__ = version("kavier")
except PackageNotFoundError:  # source tree without dist metadata
    __version__ = "0.0.0+unknown"
