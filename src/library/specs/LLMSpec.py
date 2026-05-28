from __future__ import annotations


class LLMSpec:
    """LLM specification consumed by the Kavier physics engine.

    Sparse-architecture models (formerly MoE) are encoded by setting
    ``active_params`` to the per-token active subset; the engine reasons over
    that single number and does not track expert structure explicitly."""

    def __init__(
        self,
        llm_name: str,
        n_layers: int,
        d_model: int,
        p_bytes: int,
        m_params: float,
        n_heads: int,
        d_head: int,
        active_params: float | None = None,
    ):
        self.name = llm_name
        self.d_model = d_model
        self.m_params = m_params
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.p_bytes = p_bytes
        self.active_params = active_params if active_params is not None else m_params
