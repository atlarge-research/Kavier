"""Single home for default values duplicated across the CLI, the UI, and the SDK facades.

Import-light by contract: the only non-stdlib imports are the Phase-4 ``StrEnum`` vocabularies
(themselves stdlib-only), so import-light consumers (the inference facade, the CLI/UI arg builders)
can reference these without pulling pandas/numpy. Every value here is byte-identical to the literal it
replaces — changing one changes it at every reference at once.
"""

from __future__ import annotations

from kavier.sdk.inference.core.config import CacheAction, CacheScope

# --- catalog defaults: the model/GPU a bare inference/training invocation falls back to -------------
DEFAULT_INFERENCE_MODEL = "Llama-3-8B"
DEFAULT_INFERENCE_GPU = "A10"
DEFAULT_TRAINING_MODEL = "mistral-7b-v0.1"
DEFAULT_TRAINING_GPU = "NVIDIA-A100-SXM4-80GB"

# --- inference workload defaults --------------------------------------------------------------------
DEFAULT_PREFIX_MIN_TOKENS = 1024
DEFAULT_EXPORT_RATE = 0.1  # state-snapshot interval, in seconds
DEFAULT_CACHE_SCOPE = CacheScope.SESSION

# Prefix-cache policy: the CLI/UI default (skip prefill on a hit) and the facade default (cache inert)
# diverge ON PURPOSE — the facade's synthetic workload shares no prompt content, so it opts out. Homed
# as two DISTINCTLY-NAMED constants so the split stays explicit; do NOT unify their values.
DEFAULT_CLI_PREFIX_POLICY = CacheAction.PREFILL
DEFAULT_FACADE_PREFIX_POLICY = CacheAction.NONE

# --- carbon / cost defaults (shared by the inference and training facades) --------------------------
DEFAULT_INTENSITY_G_KWH = 400.0
DEFAULT_GPU_HOUR_PRICE = 2.5
