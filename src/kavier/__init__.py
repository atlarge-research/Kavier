from __future__ import annotations

import importlib as _importlib
import sys as _sys
from importlib.abc import Loader as _Loader
from importlib.abc import MetaPathFinder as _MetaPathFinder
from importlib.machinery import ModuleSpec as _ModuleSpec
from types import ModuleType as _ModuleType
from typing import Any as _Any
from typing import Sequence as _Sequence

# Map ``kavier.<alias>`` -> underlying top-level package name.
_ALIAS_TO_LEGACY = {
    "training": "kavier_training",
    "inference": "kavier_inference",
    "io": "kavier_io",
    "energy": "kavier_energy",
    "co2": "kavier_co2",
    "library": "kavier_library",
    "opendc": "kavier_opendc",
}


class _LegacyAliasFinder(_MetaPathFinder):
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
    legacy = _ALIAS_TO_LEGACY.get(name)
    if legacy is not None:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Public API re-exports (canonical surface) ---------------------------- #
from kavier_training.core.engine import (  # noqa: E402
    simulate_full_training,
    simulate_training_step,
)
from kavier_library.gpu import GPU_SPEC_LIBRARY  # noqa: E402
from kavier_library.llm import LLM_SPEC_LIBRARY  # noqa: E402

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
