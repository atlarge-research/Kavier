"""Friendly lookups for the GPU and LLM spec libraries.

The raw ``GPU_SPEC_LIBRARY`` / ``LLM_SPEC_LIBRARY`` dicts are exported unchanged
(Coastline and other consumers index them directly). These helpers add a
human-friendly failure mode for the *entry-point* boundary: an unknown key
raises an error that names the bad key **and** lists what is available, instead
of a bare ``KeyError('foo')`` traceback.

The raised ``UnknownSpecError`` subclasses ``KeyError`` so any existing
``except KeyError`` handling keeps working.
"""

from __future__ import annotations

from library.gpu import GPU_SPEC_LIBRARY
from library.llm import LLM_SPEC_LIBRARY
from library.specs.GPUSpec import GPUSpec
from library.specs.LLMSpec import LLMSpec

__all__ = ["UnknownSpecError", "get_gpu", "get_llm"]


class UnknownSpecError(KeyError):
    """Raised when a model/GPU name is not in its spec library.

    Subclasses ``KeyError`` so callers that catch ``KeyError`` still work, but
    ``str(exc)`` carries a readable, actionable message.
    """

    def __init__(self, kind: str, name: str, available: list[str]) -> None:
        self._message = f"Unknown {kind} {name!r}. Available {kind}s ({len(available)}): {', '.join(available)}"
        super().__init__(self._message)

    def __str__(self) -> str:  # KeyError.__str__ would re-quote the message
        return self._message


def get_gpu(name: str) -> GPUSpec:
    """Return the :class:`GPUSpec` for ``name`` or raise :class:`UnknownSpecError`."""
    if name not in GPU_SPEC_LIBRARY:
        raise UnknownSpecError("GPU", name, sorted(GPU_SPEC_LIBRARY))
    return GPU_SPEC_LIBRARY[name]


def get_llm(name: str) -> LLMSpec:
    """Return the :class:`LLMSpec` for ``name`` or raise :class:`UnknownSpecError`."""
    if name not in LLM_SPEC_LIBRARY:
        raise UnknownSpecError("model", name, sorted(LLM_SPEC_LIBRARY))
    return LLM_SPEC_LIBRARY[name]
