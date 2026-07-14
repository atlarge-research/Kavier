"""One lazy-export helper for the package ``__init__`` modules (behaviour-preserving dedup).

Several packages re-export names lazily via a module-level PEP 562 ``__getattr__`` so a bare
``import`` stays cheap and stdlib-light — no pandas/numpy/matplotlib (nor any engine) is pulled in
until a name is first accessed. This module factors out that one repeated pattern.

Naming convention (so internal code stops mixing the two): the public entrypoint for a domain is
``kavier.X`` (e.g. ``kavier.inference``, ``kavier.cluster``); the implementation lives at
``kavier.sdk.X``. ``kavier.X`` is only a thin lazy alias — internal code should import the
``kavier.sdk.X`` implementation directly.

Stdlib-only by contract (``importlib`` + ``typing``): importing this module must not pull any heavy
dependency, so ``import kavier`` and ``import kavier.sdk.cluster`` stay import-light.
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
    """Build a PEP 562 module ``__getattr__`` that resolves lazy exports on first access.

    ``module_globals`` is the caller's live ``globals()`` — used both to read the package
    ``__name__`` and to cache each resolved value back into it, so subsequent access skips
    ``__getattr__``. Targets are submodule paths *relative to the caller package*: value ``"facade"``
    for package ``kavier.sdk.inference`` resolves ``kavier.sdk.inference.facade`` (dotted values such
    as ``"sdk.library"`` work too).

    - ``modules`` maps an exported name to a submodule whose *module object* is returned (aliases).
    - ``attrs`` maps an exported name to the submodule it is an *attribute* of (``getattr`` is
      returned).

    An unknown name raises the standard
    ``AttributeError(f"module {__name__!r} has no attribute {name!r}")``.
    """
    package: str = module_globals["__name__"]
    module_map: Mapping[str, str] = modules or {}
    attr_map: Mapping[str, str] = attrs or {}

    def __getattr__(name: str) -> Any:
        submodule = module_map.get(name)
        if submodule is not None:
            module = importlib.import_module(f"{package}.{submodule}")
            module_globals[name] = module  # cache so subsequent access skips __getattr__
            return module
        source = attr_map.get(name)
        if source is not None:
            value = getattr(importlib.import_module(f"{package}.{source}"), name)
            module_globals[name] = value  # cache so subsequent access skips __getattr__
            return value
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    return __getattr__
