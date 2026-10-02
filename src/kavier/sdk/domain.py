"""Simulation-domain names shared by the CLI dispatcher and the carbon facades.

``StrEnum`` members are ``str``, so a member equals its value as a registry key or serialised output.
Stdlib-only: ``kavier/__init__.py`` imports this module at load time.
"""

from __future__ import annotations

from enum import StrEnum


class Domain(StrEnum):
    """A simulator the facades can run: inference or training."""

    INFERENCE = "inference"
    TRAINING = "training"


#: Key for the producing simulator in a carbon-billing result.
#: Not one of the frozen ``performance()`` columns.
RESULT_SOURCE_KEY = "source"
