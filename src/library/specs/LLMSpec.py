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
        """
        Initialize the LLMSpec with model specifications.

        Parameters:
        - llm_name (str): Name of the model.
        - n_layers (int): Number of transformer layers in the model.
        - d_model (int): Model hidden dimension size.
        - p_bytes (int): Number of bytes per parameter (e.g., 2 for FP16).
        - m_params (float): Total number of model parameters.
        - n_heads (int): Number of attention heads.
        - d_head (int): Dimension per attention head.
        - active_params (float | None): Parameters active per forward pass.
            For MoE models, only a subset of experts are active per token.
            Defaults to m_params (dense models).
        - num_experts (int): Total number of experts (1 = dense model).
        - active_experts (int): Experts active per token (for MoE routing).
        """
        self.name = llm_name
        self.d_model = d_model
        self.m_params = m_params
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.p_bytes = p_bytes
        self.num_experts = num_experts
        self.active_experts = active_experts
        # For FLOPs: use active_params (MoE: only active experts contribute)
        self.active_params = active_params if active_params is not None else m_params

    @property
    def is_moe(self) -> bool:
        """Whether this is a Mixture-of-Experts model."""
        return self.num_experts > 1
