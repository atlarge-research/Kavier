"""Name-based lookups into the GPU and LLM spec libraries."""

from __future__ import annotations

from kavier_library.gpu import GPU_SPEC_LIBRARY
from kavier_library.llm import LLM_SPEC_LIBRARY
from kavier_library.specs.GPUSpec import GPUSpec
from kavier_library.specs.LLMSpec import LLMSpec

__all__ = ["UnknownSpecError", "get_gpu", "get_llm"]


class UnknownSpecError(KeyError):
    """Raised when a GPU/model name is absent; message lists available names."""

    def __init__(self, kind: str, name: str, available: list[str]) -> None:
        self._message = f"Unknown {kind} {name!r}. Available {kind}s ({len(available)}): {', '.join(available)}"
        super().__init__(self._message)

    def __str__(self) -> str:  # KeyError.__str__ would re-quote the message
        return self._message


def get_gpu(name: str) -> GPUSpec:
    if name not in GPU_SPEC_LIBRARY:
        raise UnknownSpecError("GPU", name, sorted(GPU_SPEC_LIBRARY))
    return GPU_SPEC_LIBRARY[name]


def get_llm(name: str) -> LLMSpec:
    if name not in LLM_SPEC_LIBRARY:
        raise UnknownSpecError("model", name, sorted(LLM_SPEC_LIBRARY))
    return LLM_SPEC_LIBRARY[name]
