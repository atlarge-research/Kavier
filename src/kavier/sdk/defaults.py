"""Default values shared by the CLI and the SDK facades.

Imports only the stdlib-only ``StrEnum`` vocabularies, so the inference facade and the CLI arg
builders can use these values without loading pandas or numpy.
"""

from __future__ import annotations

from kavier.sdk.inference.core.config import CacheAction, CacheScope

# Model and GPU used when an inference or training run names none.
DEFAULT_INFERENCE_MODEL = "Llama-3-8B"
DEFAULT_INFERENCE_GPU = "A10"
DEFAULT_TRAINING_MODEL = "mistral-7b-v0.1"
DEFAULT_TRAINING_GPU = "NVIDIA-A100-SXM4-80GB"

# Inference workload.
DEFAULT_PREFIX_MIN_TOKENS = 1024
DEFAULT_EXPORT_RATE = 0.1  # state-snapshot interval, in seconds
DEFAULT_CACHE_SCOPE = CacheScope.SESSION

# Prefix-cache policy. The CLI/UI skip prefill on a hit; the facade leaves the cache inert because its
# synthetic workload shares no prompt content. The two values differ and should stay separate.
DEFAULT_CLI_PREFIX_POLICY = CacheAction.PREFILL
DEFAULT_FACADE_PREFIX_POLICY = CacheAction.NONE

# Carbon and cost, shared by the inference and training facades.
DEFAULT_INTENSITY_G_KWH = 400.0
DEFAULT_GPU_HOUR_PRICE = 2.5
