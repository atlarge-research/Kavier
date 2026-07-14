"""Shared unit-conversion constants: seconds/hour, watt-seconds/kWh, Wh/kWh, tokens/Mtoken, ms/second,
grams/kg, and FLOPs/TFLOP.

Stdlib-only (no imports beyond ``__future__``) — imported by :mod:`kavier.sdk.cluster.facade`, whose
import-light contract (``tests/test_cluster/test_import_light.py``) forbids pulling in pandas/numpy.
"""

from __future__ import annotations

SECONDS_PER_HOUR = 3600.0
WS_PER_KWH = 3.6e6  # watt-seconds per kWh
WH_PER_KWH = 1000.0
TOKENS_PER_MTOKEN = 1_000_000.0
MS_PER_SECOND = 1000.0
G_PER_KG = 1000.0  # grams per kilogram
FLOPS_PER_TFLOP = 1e12  # floating-point ops per teraflop


def per_mtoken(value: float, total_tokens: float) -> float:
    """``value`` scaled to a per-million-token rate; ``0.0`` when ``total_tokens`` is falsy.

    Preserves the ``value * (TOKENS_PER_MTOKEN / total_tokens)`` operation order (division before
    multiplication) used at every call site so results stay bit-identical to the pre-refactor code.
    """
    return value * (TOKENS_PER_MTOKEN / total_tokens) if total_tokens else 0.0
