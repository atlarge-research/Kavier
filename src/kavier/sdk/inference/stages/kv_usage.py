"""Analytical KV-cache memory-utilization model over a request's prefill/decode timeline."""

from kavier.sdk.library.specs.GPUSpec import GPUSpec
from kavier.sdk.library.specs.LLMSpec import LLMSpec


def get_kv_cache_utilization(
    llm: LLMSpec, gpu: GPUSpec, t_prefill, t_decode, t, prompt_len, response_len, kv_cache
) -> float:
    """Return the fraction of GPU memory occupied by the KV cache at elapsed time ``t`` (s); 0 when KV cache is off."""
    if not kv_cache:
        return 0

    bytes_per_token = llm.n_layers * llm.d_model * 2 * llm.p_bytes

    if t <= t_prefill:
        tokens = prompt_len * (t / t_prefill) if t_prefill > 0 else prompt_len
    else:
        tokens = prompt_len
        elapsed_decode = min(t - t_prefill, t_decode)
        if t_decode > 0:
            tokens += response_len * (elapsed_decode / t_decode)

    used_bytes = tokens * bytes_per_token
    total_bytes = gpu.memory_gb * 1024**3
    return used_bytes / total_bytes
