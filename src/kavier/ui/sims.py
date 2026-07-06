"""kavier.ui adapter: re-exports the sim wrappers from ``kavier.sdk.inference`` / ``kavier.sdk.training``."""

from __future__ import annotations

from kavier.sdk.inference import (
    energy_from_inference,
    export_opendc,
    run_carbon_from_inference,
    run_inference,
)
from kavier.sdk.library import GPU_SPEC_LIBRARY, LLM_SPEC_LIBRARY
from kavier.sdk.training import run_carbon_from_training, run_training

__all__ = [
    "GPU_SPEC_LIBRARY",
    "LLM_SPEC_LIBRARY",
    "model_names",
    "gpu_names",
    "run_inference",
    "run_training",
    "run_carbon_from_inference",
    "run_carbon_from_training",
    "energy_from_inference",
    "export_opendc",
]


def model_names() -> list[str]:
    """Sorted list of available LLM names."""
    return sorted(LLM_SPEC_LIBRARY)


def gpu_names() -> list[str]:
    """Sorted list of available GPU names."""
    return sorted(GPU_SPEC_LIBRARY)
