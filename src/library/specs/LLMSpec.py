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
        """
        self.name = llm_name
        self.d_model = d_model
        self.m_params = m_params
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.p_bytes = p_bytes
