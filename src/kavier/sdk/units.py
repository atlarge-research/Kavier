"""Unit-conversion constants.

Stdlib-only: :mod:`kavier.sdk.cluster.facade` imports this module and must not load pandas or numpy
(``tests/test_cluster/test_import_light.py``).
"""

from __future__ import annotations

SECONDS_PER_HOUR = 3600.0
WS_PER_KWH = 3.6e6  # watt-seconds per kWh
WH_PER_KWH = 1000.0
TOKENS_PER_MTOKEN = 1_000_000.0
MS_PER_SECOND = 1000.0
G_PER_KG = 1000.0
FLOPS_PER_TFLOP = 1e12


def per_mtoken(value: float, total_tokens: float) -> float:
    """Return ``value`` per million tokens, or ``0.0`` when ``total_tokens`` is falsy.

    Divides before multiplying, as every call site does, so results are bit-identical.
    """
    return value * (TOKENS_PER_MTOKEN / total_tokens) if total_tokens else 0.0
