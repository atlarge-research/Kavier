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
