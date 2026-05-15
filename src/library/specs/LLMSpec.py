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
        num_experts: int = 1,
        active_experts: int = 1,
    ):
        self.name = llm_name
        self.d_model = d_model
        self.m_params = m_params
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.p_bytes = p_bytes
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.active_params = active_params if active_params is not None else m_params

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 1
