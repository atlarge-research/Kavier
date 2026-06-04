"""Kavier — canonical top-level package.

This module is the *canonical* import surface for new consumers::

    import kavier
    from kavier import simulate_training_step, simulate_full_training
    from kavier import GPU_SPEC_LIBRARY, LLM_SPEC_LIBRARY

Historical layout
-----------------
Kavier originally shipped its code as a set of *sibling* top-level packages
under ``src/`` (``kavier_training``, ``kavier_inference``, ``kavier_io``,
``kavier_energy``, ``kavier_co2``, ``library``, ``opendc``). Those packages are
still the physical, on-disk layout and remain fully importable under their
original names — the sibling Coastline project links against them by exact
path via ``PYTHONPATH=kavier/src`` and must keep working unchanged. We do NOT
move them; this ``kavier`` package is a thin, additive re-export layer on top.

What this package provides
--------------------------
* The public API re-exported at the top level (see ``__all__``).
* Convenience submodule aliases so new code can spell the legacy packages with
  a ``kavier.`` prefix, e.g. ``kavier.training`` -> ``kavier_training``,
  ``kavier.inference`` -> ``kavier_inference``, ``kavier.io`` -> ``kavier_io``,
  ``kavier.energy`` -> ``kavier_energy``, ``kavier.co2`` -> ``kavier_co2``,
  ``kavier.library`` -> ``library``, ``kavier.opendc`` -> ``opendc``.

  These aliases resolve to the *same* module objects as the legacy packages,
  including deep submodules: ``from kavier.training.core.engine import
  simulate_training_step`` returns the exact same function object as
  ``from kavier_training.core.engine import simulate_training_step``, and
  ``kavier.library.gpu.GPU_SPEC_LIBRARY is library.gpu.GPU_SPEC_LIBRARY``. A
  meta-path finder (installed below) redirects any ``kavier.<alias>...`` import
  to its legacy counterpart, so there is never a duplicate module/dict — this
  keeps the live ``calibration._CAL`` swap contract intact regardless of which
  spelling a consumer uses.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys
from importlib.abc import Loader as _Loader
from importlib.abc import MetaPathFinder as _MetaPathFinder
from importlib.machinery import ModuleSpec as _ModuleSpec
from types import ModuleType as _ModuleType
from typing import Any as _Any
from typing import Sequence as _Sequence

# Map ``kavier.<alias>`` -> legacy top-level package name.
_ALIAS_TO_LEGACY = {
    "training": "kavier_training",
    "inference": "kavier_inference",
    "io": "kavier_io",
    "energy": "kavier_energy",
    "co2": "kavier_co2",
    "library": "library",
    "opendc": "opendc",
}


class _LegacyAliasFinder(_MetaPathFinder):
    """Redirect ``kavier.<alias>`` (and deep submodules) to the legacy module.

    For an import of e.g. ``kavier.training.core.engine`` we import the legacy
    ``kavier_training.core.engine`` and publish the *same* module object under
    the ``kavier.`` name, so both spellings share one module instance.
    """

    _prefix = f"{__name__}."

    def find_spec(
        self,
        fullname: str,
        path: _Sequence[str] | None = None,
        target: _ModuleType | None = None,
    ) -> _ModuleSpec | None:
        if not fullname.startswith(self._prefix):
            return None
        tail = fullname[len(self._prefix) :]
        head, _, rest = tail.partition(".")
        legacy_root = _ALIAS_TO_LEGACY.get(head)
        if legacy_root is None:
            return None
        legacy_name = legacy_root if not rest else f"{legacy_root}.{rest}"
        spec = _ModuleSpec(fullname, _LegacyAliasLoader(legacy_name))
        return spec


class _LegacyAliasLoader(_Loader):
    def __init__(self, legacy_name: str) -> None:
        self._legacy_name = legacy_name

    def create_module(self, spec: _ModuleSpec) -> _ModuleType:
        module = _importlib.import_module(self._legacy_name)
        # Register under the requested ``kavier.`` name too (idempotent).
        _sys.modules[spec.name] = module
        return module

    def exec_module(self, module: _ModuleType) -> None:  # already executed
        return None


_sys.meta_path.insert(0, _LegacyAliasFinder())


def __getattr__(name: str) -> _Any:
    """Lazily expose ``kavier.<alias>`` as an attribute (PEP 562)."""
    legacy = _ALIAS_TO_LEGACY.get(name)
    if legacy is not None:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Public API re-exports (canonical surface) ---------------------------- #
from kavier_training.core.engine import (  # noqa: E402
    simulate_full_training,
    simulate_training_step,
)
from library.gpu import GPU_SPEC_LIBRARY  # noqa: E402
from library.llm import LLM_SPEC_LIBRARY  # noqa: E402

__all__ = [
    "simulate_training_step",
    "simulate_full_training",
    "GPU_SPEC_LIBRARY",
    "LLM_SPEC_LIBRARY",
    "training",
    "inference",
    "io",
    "energy",
    "co2",
    "library",
    "opendc",
]

__version__ = "0.3.0"
