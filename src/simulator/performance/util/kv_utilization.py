from simulator.performance.util.specs import GPUSpec, LLMSpec


def get_kv_cache_utilization(
        llm: LLMSpec, gpu: GPUSpec, t_prefill, t_decode, t, prompt_len, response_len, kv_cache
) -> float:
    """
    KV-cache usage (%) of GPU VRAM at time t.
    :param llm: LLMSpec, the LLM specification, needs .n_layers, d_model, p_bytes.
    :param gpu: GPUSpec, the GPU specification, needs .memory_gb.
    :param t_prefill: time taken for prefill phase in seconds.
    :param t_decode: time taken for decode phase in seconds.
    :param t: current time in seconds.
    :return:
    """
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
    total_bytes = gpu.memory_gb * 1024 ** 3
    total_byte = total_bytes
    return used_bytes / total_byte
