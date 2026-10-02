"""Simulation-domain names shared by the CLI dispatcher, the UI, and the carbon facades.

``StrEnum`` members are ``str``, so a member equals its value as a registry key, menu value, or
serialised output. Stdlib-only: ``kavier/__init__.py`` imports this module at load time.
"""

from __future__ import annotations

from enum import StrEnum


class Domain(StrEnum):
    """A simulator the facades and UI can run: inference or training."""

    INFERENCE = "inference"
    TRAINING = "training"


#: Key for the producing simulator in a carbon-billing result and the UI "Source" field.
#: Not one of the frozen ``performance()`` columns.
RESULT_SOURCE_KEY = "source"
