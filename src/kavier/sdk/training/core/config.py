"""Training-engine configuration: the fine-tuning ``Method`` vocabulary + infrastructure defaults."""

from __future__ import annotations

from enum import StrEnum


class Method(StrEnum):
    """Fine-tuning method. ``StrEnum`` members are ``str``, so ``Method.FULL == "full"`` and every
    boundary that passes a plain method string (Coastline, CSV rows, the CLI) keeps working."""

    FULL = "full"
    LORA = "lora"
    GPTQ_LORA = "gptq-lora"


# Inter-node interconnect (Gbps): 200 = HDR InfiniBand (100 = EDR, 400 = NDR).
INFINIBAND_GBPS = 200.0

# Ring-all-reduce cost model (data-parallel gradient synchronisation).
RING_ALLREDUCE_LATENCY_S = 5e-6  # per-hop link latency
RING_ALLREDUCE_OVERHEAD_PER_MSG_S = 2e-6  # fixed per-message overhead
BITS_PER_BYTE = 8  # divisor turning a Gbps link rate into GB/s
