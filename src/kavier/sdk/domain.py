"""Simulation-domain vocabulary shared by the CLI dispatcher, the UI, and the carbon facades.

``Domain`` names the two batch-predictor simulators (``inference`` / ``training``); ``StrEnum``
members are ``str``, so a member is byte-identical to its value everywhere it is used as a registry
key, menu value, or serialised output. Stdlib-only by contract: ``kavier/__init__.py`` imports this
at module load, so it must never pull anything heavier than ``enum``.
"""

from __future__ import annotations

from enum import StrEnum


class Domain(StrEnum):
    """A simulator the facades and UI can run: inference or training."""

    INFERENCE = "inference"
    TRAINING = "training"


#: Output-dict key naming the producing simulator in a carbon-billing result (and the UI's
#: rendered "Source" field). Not a frozen ``performance()`` column.
RESULT_SOURCE_KEY = "source"
