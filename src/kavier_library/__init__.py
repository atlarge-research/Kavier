"""Kavier spec library: GPU/LLM catalogues and name-based lookups."""

from .gpu import GPU_SPEC_LIBRARY
from .llm import LLM_SPEC_LIBRARY
from .lookup import UnknownSpecError, get_gpu, get_llm

__all__ = [
    "GPU_SPEC_LIBRARY",
    "LLM_SPEC_LIBRARY",
    "UnknownSpecError",
    "get_gpu",
    "get_llm",
]
