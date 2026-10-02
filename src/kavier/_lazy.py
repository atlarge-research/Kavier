"""Lazy exports for package ``__init__`` modules via a PEP 562 ``__getattr__``.

Packages use it to defer pandas, numpy, matplotlib and the engines until a name is first accessed.

The public name for a domain is ``kavier.X`` (e.g. ``kavier.inference``) and its implementation is
``kavier.sdk.X``. Internal code imports ``kavier.sdk.X`` directly.

Stdlib-only, so ``import kavier`` and ``import kavier.sdk.cluster`` stay light.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping


def lazy_getattr(
    module_globals: dict[str, Any],
    *,
    modules: Mapping[str, str] | None = None,
    attrs: Mapping[str, str] | None = None,
) -> Callable[[str], Any]:
    """Return a PEP 562 module ``__getattr__`` that resolves lazy exports on first access.

    Args:
        module_globals: The caller's ``globals()``. Supplies the package name and caches each
            resolved value, so later lookups bypass ``__getattr__``.
        modules: Exported name -> submodule; the module object is returned.
        attrs: Exported name -> submodule that defines it; the attribute is returned.

    Submodule paths are relative to the caller package and may be dotted: ``"facade"`` in
    ``kavier.sdk.inference`` resolves ``kavier.sdk.inference.facade``. An unknown name raises the
    standard ``AttributeError``.
    """
    package: str = module_globals["__name__"]
    module_map: Mapping[str, str] = modules or {}
    attr_map: Mapping[str, str] = attrs or {}

    def __getattr__(name: str) -> Any:
        submodule = module_map.get(name)
        if submodule is not None:
            module = importlib.import_module(f"{package}.{submodule}")
            module_globals[name] = module
            return module
        source = attr_map.get(name)
        if source is not None:
            value = getattr(importlib.import_module(f"{package}.{source}"), name)
            module_globals[name] = value
            return value
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    return __getattr__
