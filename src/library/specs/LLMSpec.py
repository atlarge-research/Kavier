from __future__ import annotations


class LLMSpec:
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
