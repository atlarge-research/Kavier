"""Fine-tuning ``Method`` names and interconnect constants for the training engine."""

from __future__ import annotations

from enum import StrEnum


class Method(StrEnum):
    """Fine-tuning method.

    Members are ``str``, so ``Method.FULL == "full"`` and plain method strings from Coastline, CSV rows
    and the CLI compare equal.
    """

    FULL = "full"
    LORA = "lora"
    GPTQ_LORA = "gptq-lora"


# Inter-node interconnect [Gbps]: HDR InfiniBand (EDR = 100, NDR = 400).
INFINIBAND_GBPS = 200.0

# Ring all-reduce cost model for data-parallel gradient sync.
RING_ALLREDUCE_LATENCY_S = 5e-6  # per-hop link latency
RING_ALLREDUCE_OVERHEAD_PER_MSG_S = 2e-6  # fixed per-message overhead
BITS_PER_BYTE = 8  # Gbps -> GB/s
