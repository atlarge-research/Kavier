"""
MFU (Model FLOPs Utilization) Calculator Module

This module provides utilities for calculating hardware and computational MFU metrics
for large language model training runs. It supports various model architectures,
fine-tuning methods (full, LoRA, GPTQ-LoRA), and GPU configurations.

Key Metrics:
- MFU_hardware: Fraction of peak hardware FLOPs achieved (method-agnostic)
- MFU_computational: Efficiency of method-specific compute (accounts for LoRA, quantization)

References:
- Chowdhery et al. (2022) "PaLM: Scaling Language Modeling with Pathways"
- Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
- Frantar et al. (2022) "GPTQ: Accurate Post-Training Quantization"
- Kaplan et al. (2020) "Scaling Laws for Neural Language Models"
"""

import re

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

# Model configs — all values are hardcoded (no HF calls at runtime).
# Sources:
#   - Confirmed via HF safetensors metadata where noted
#   - Computed from architecture arithmetic otherwise (marked with ~)
#   - MoE models carry both num_parameters (total) and active_parameters (per-token)
_model_configs = {
    # ── Granite legacy ────────────────────────────────────────────────────────
    "granite-7b-base": {  # HF safetensors confirmed
        "num_parameters": 6_738_415_616,
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    },
    "granite-8b-code-base": {  # HF safetensors confirmed
        "num_parameters": 8_054_910_976,
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "vocab_size": 49152,
    },
    "granite-3b-code-base-128k": {  # HF safetensors confirmed
        "num_parameters": 3_482_503_680,
        "num_hidden_layers": 32,
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "vocab_size": 49152,
    },
    "granite-34b-code-base": {  # HF safetensors confirmed
        "num_parameters": 33_715_603_456,
        "num_hidden_layers": 88,
        "hidden_size": 6144,
        "intermediate_size": None,  # GPT-BigCode style (no SwiGLU split)
        "vocab_size": 49152,
    },
    "granite-20b-v2": {  # HF safetensors confirmed
        "num_parameters": 20_066_993_152,
        "num_hidden_layers": 52,
        "hidden_size": 6144,
        "intermediate_size": None,  # GPT-BigCode style
        "vocab_size": 49152,
    },
    "granite-13b-v2": {  # private; ~LLaMA-13B architecture
        "num_parameters": 13_000_000_000,
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "vocab_size": 32000,
    },
    "granite-8b-japanese": {  # private; ~granite-3.x-8B architecture
        "num_parameters": 8_171_000_000,
        "num_hidden_layers": 40,
        "hidden_size": 4096,
        "intermediate_size": 12800,
        "vocab_size": 49152,
    },
    # ── Granite 3.x ───────────────────────────────────────────────────────────
    "granite-3-8b": {  # ~ same arch as granite-3.3-8b
        "num_parameters": 8_170_864_640,
        "num_hidden_layers": 40,
        "hidden_size": 4096,
        "intermediate_size": 12800,
        "vocab_size": 49155,
    },
    "granite-3.1-2b": {  # ~ computed from AutoConfig
        "num_parameters": 2_530_000_000,
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 49155,
    },
    "granite-3.1-2b-instruct": {  # ~ computed from AutoConfig
        "num_parameters": 2_530_000_000,
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 49155,
    },
    "granite-2b-base": {  # ~ computed from AutoConfig
        "num_parameters": 2_530_000_000,
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 49155,
    },
    "granite-3.1-3b-a800m-instruct": {  # MoE: 40 experts, 8 active per token
        "num_parameters": 3_370_000_000,  # total (all expert weights)
        "active_parameters": 800_000_000,  # active per token (~800M); used for MFU_hardware
        "num_hidden_layers": 32,
        "hidden_size": 1536,
        "intermediate_size": 512,  # per-expert FFN size
        "num_local_experts": 40,
        "num_experts_per_tok": 8,
        "vocab_size": 49155,
    },
    "granite-3.1-8b-instruct": {  # ~ same arch as granite-3.3-8b
        "num_parameters": 8_170_864_640,
        "num_hidden_layers": 40,
        "hidden_size": 4096,
        "intermediate_size": 12800,
        "vocab_size": 49155,
    },
    "granite-3.1-8b-base": {  # ~ same arch as granite-3.3-8b
        "num_parameters": 8_170_864_640,
        "num_hidden_layers": 40,
        "hidden_size": 4096,
        "intermediate_size": 12800,
        "vocab_size": 49155,
    },
    "granite-3.3-8b": {  # HF safetensors confirmed
        "num_parameters": 8_170_864_640,
        "num_hidden_layers": 40,
        "hidden_size": 4096,
        "intermediate_size": 12800,
        "vocab_size": 49159,
    },
    "granite-3.3-8b-instruct": {  # HF safetensors confirmed
        "num_parameters": 8_170_864_640,
        "num_hidden_layers": 40,
        "hidden_size": 4096,
        "intermediate_size": 12800,
        "vocab_size": 49159,
    },
    # ── Granite 4.0 (HF safetensors confirmed) ────────────────────────────────
    "granite-4.0-350m": {  # ibm-granite/granite-4.0-350m
        "num_parameters": 440_401_920,
        "num_hidden_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "vocab_size": 100352,
    },
    "granite-4.0-1b": {  # ibm-granite/granite-4.0-1b
        "num_parameters": 1_753_219_072,
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "intermediate_size": 4096,
        "vocab_size": 100352,
    },
    "granite-4.0-micro": {  # ibm-granite/granite-4.0-micro
        "num_parameters": 3_240_099_840,
        "num_hidden_layers": 40,
        "hidden_size": 2560,
        "intermediate_size": 8192,
        "vocab_size": 100352,
    },
    "granite-4.0-h-tiny": {  # ibm-granite/granite-4.0-h-tiny
        "num_parameters": 748_683_264,
        "num_hidden_layers": 40,
        "hidden_size": 1536,
        "intermediate_size": 512,
        "vocab_size": 100352,
    },
    "granite-4.0-h-micro": {  # ibm-granite/granite-4.0-h-micro
        "num_parameters": 2_424_307_712,
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 100352,
    },
    "granite-4.0-h-1b": {
        "num_parameters": 1_753_219_072,
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "intermediate_size": 4096,
        "vocab_size": 100352,
    },
    "granite-4.0-h-small": {  # ibm-granite/granite-4.0-h-small
        "num_parameters": 3_758_096_384,
        "num_hidden_layers": 40,
        "hidden_size": 4096,
        "intermediate_size": 768,
        "vocab_size": 100352,
    },
    # ── ALLaM ─────────────────────────────────────────────────────────────────
    "allam-1-13b": {  # private; LLaMA-1-13B architecture
        "num_parameters": 13_015_864_320,
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "vocab_size": 32000,
    },
    # ── LLaMA family (gated on HF; values from published model cards) ─────────
    "llama-7b": {
        "num_parameters": 6_738_415_616,
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    },
    "llama-13b": {
        "num_parameters": 13_015_864_320,
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "vocab_size": 32000,
    },
    "llama2-70b": {
        "num_parameters": 68_976_648_192,
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "vocab_size": 32000,
    },
    "llama3-8b": {
        "num_parameters": 8_030_261_248,
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "vocab_size": 128256,
    },
    "llama3-70b": {
        "num_parameters": 70_553_706_496,
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "vocab_size": 128256,
    },
    "llama3.1-8b": {
        "num_parameters": 8_030_261_248,
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "vocab_size": 128256,
    },
    "llama3.1-70b": {
        "num_parameters": 70_553_706_496,
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "vocab_size": 128256,
    },
    "llama3.1-405b": {
        "num_parameters": 404_650_311_680,
        "num_hidden_layers": 126,
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "vocab_size": 128256,
    },
    "llama3.2-1b": {
        "num_parameters": 1_235_814_400,
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 128256,
    },
    "llama3.2-3b": {
        "num_parameters": 3_212_749_824,
        "num_hidden_layers": 28,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "vocab_size": 128256,
    },
    # ── Mistral / Mixtral ─────────────────────────────────────────────────────
    "mistral-7b-v0.1": {  # HF safetensors confirmed
        "num_parameters": 7_241_732_096,
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "vocab_size": 32000,
    },
    "mixtral-8x7b-instruct-v0.1": {  # HF safetensors confirmed; MoE
        "num_parameters": 46_702_792_704,  # total (all expert weights)
        "active_parameters": 12_888_080_384,  # 2 of 8 experts active per token
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "vocab_size": 32000,
    },
    "mistral-123b-v2": {  # Mistral Large 2 (2407); ~ from architecture
        "num_parameters": 123_522_109_440,
        "num_hidden_layers": 88,
        "hidden_size": 12288,
        "intermediate_size": 28672,
        "vocab_size": 32768,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# GPU SPECIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

# Peak FLOPs for GPUs in BF16 (from NVIDIA datasheets)
# A100 SXM4-80GB  : 312 TFLOPS BF16  (techpowerup c3746)
# A100 PCIe-80GB  : 312 TFLOPS BF16  (techpowerup c3821)
# L40S            : 362.05 TFLOPS BF16
gpu_flops = {
    "L40S": 362.05e12,
    "A100": 312e12,  # covers both SXM4 and PCIe 80GB variants
    "NVIDIA-A100-SXM4-80GB": 312e12,
    "NVIDIA-A100-80GB-PCIe": 312e12,
    "NVIDIA A100-SXM4-80GB": 312e12,
    "H100": 989.5e12,
}

_GPU_ALIASES = {
    "a100": "A100",
    "l40s": "L40S",
    "h100": "H100",
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def get_model_config(model_name):
    """
    Retrieve model configuration dictionary.

    Args:
        model_name: Model name string (case-insensitive)

    Returns:
        dict: Model configuration or None if not found
    """
    if pd.isna(model_name):
        return None
    return _model_configs.get(str(model_name).strip().lower())


def get_mfu_params(model_name):
    """
    Active params for MoE, total params for dense — baseline for hardware MFU.

    Args:
        model_name: Model name string

    Returns:
        int: Parameter count or np.nan if not found
    """
    cfg = get_model_config(model_name)
    if cfg is None:
        return np.nan
    p = cfg.get("active_parameters") or cfg.get("num_parameters")
    return p if p is not None else np.nan


def get_lora_params(model_name, r):
    """
    Trainable parameter count for LoRA rank r.
    Covers q, k, v, o projections and gate, up, down MLP projections.

    Args:
        model_name: Model name string
        r: LoRA rank

    Returns:
        int: LoRA trainable parameter count or np.nan if cannot compute

    Reference:
        Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
    """
    cfg = get_model_config(model_name)
    if cfg is None or pd.isna(r) or r == 0:
        return np.nan
    num_layers = cfg.get("num_hidden_layers")
    hidden_size = cfg.get("hidden_size")
    intermediate_size = cfg.get("intermediate_size")
    if any(v is None for v in [num_layers, hidden_size, intermediate_size]):
        return np.nan
    attn_params = 4 * 2 * hidden_size * r  # q, k, v, o
    mlp_params = 2 * (hidden_size + intermediate_size) * r * 2  # gate, up, down
    return num_layers * (attn_params + mlp_params)


def extract_lora_rank(experiment_id):
    """
    Extract LoRA rank r from experiment_id strings like '...r-4-a-16...'.

    Args:
        experiment_id: Experiment identifier string

    Returns:
        tuple: (r, alpha) or (None, None) if not found
    """
    m = re.search(r"r-(\d+)-a-(\d+)", str(experiment_id))
    if m:
        return int(m.group(1)), int(m.group(2))
    # Also try 'lora_r' style
    m2 = re.search(r"lora_r[_=](\d+)", str(experiment_id))
    if m2:
        return int(m2.group(1)), None
    return None, None


def get_trainable_params(row):
    """
    Effective parameter count that drives the training compute, by method:
      full      → all model parameters  (6N FLOPs/token, Kaplan et al. 2020)
      lora      → LoRA trainable params  (6·N_lora FLOPs/token, Hu et al. 2021)
      gptq-lora → same as lora           (quantization overhead added separately)

    Args:
        row: DataFrame row with 'method', 'model_name', and 'experiment_id' columns

    Returns:
        int: Trainable parameter count or np.nan
    """
    method = str(row.get("method", "")).lower().strip()
    model_name = row["model_name"]

    if method == "full":
        return get_mfu_params(model_name)

    if method in ("lora", "gptq-lora"):
        r, _ = extract_lora_rank(row.get("experiment_id", ""))
        if r is None:
            # The auto-generated code uses the experiment name to get r
            # The default value of r is 4 so we'll stick to that for now
            r = 4
        return get_lora_params(model_name, r)

    # Fallback: treat as full
    return get_mfu_params(model_name)


# ══════════════════════════════════════════════════════════════════════════════
# GPU CONFIGURATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def normalize_gpu_model(gpu_model):
    """
    Normalize GPU model name to standard format.

    Args:
        gpu_model: GPU model string

    Returns:
        str: Normalized GPU model name or original if no match
    """
    if pd.isna(gpu_model):
        return None
    s = str(gpu_model).strip()
    # Exact match first
    if s in gpu_flops:
        return s
    # Case-insensitive alias lookup
    lower = s.lower()
    for pattern, key in _GPU_ALIASES.items():
        if pattern in lower:
            return key
    return s


def _resolve_total_gpus(row):
    """
    Return total GPU count across all nodes, or fallback to number_gpus, or np.nan.

    Args:
        row: DataFrame row with 'number_gpus' and optionally 'number_nodes' columns

    Returns:
        float: Total GPU count or np.nan
    """
    number_gpus = row.get("number_gpus", np.nan)
    number_nodes = row.get("number_nodes", np.nan)

    # Try to calculate total_gpus from number_gpus * number_nodes
    if not pd.isna(number_gpus) and number_gpus > 0:
        if not pd.isna(number_nodes) and number_nodes > 0:
            return number_gpus * number_nodes
        # Fallback: if number_nodes is missing but number_gpus is valid, use number_gpus
        return number_gpus

    return np.nan


def _resolve_tokens_per_sec(row):
    """
    Return total tokens/sec across all GPUs, or np.nan.

    Args:
        row: DataFrame row with token throughput columns

    Returns:
        float: Total tokens per second or np.nan
    """
    total_gpus = _resolve_total_gpus(row)
    if pd.isna(total_gpus) or total_gpus == 0:
        return np.nan
    for col in ("dataset_tokens_per_second", "train_tokens_per_second"):
        v = row.get(col, np.nan)
        if not pd.isna(v) and v > 0:
            return v
    v = row.get("train_tokens_per_gpu_per_second", np.nan)
    if not pd.isna(v) and v > 0:
        return v * total_gpus
    return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# MFU CALCULATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def compute_mfu_hardware(row):
    """
    Hardware MFU — measures what fraction of peak hardware FLOPs are achieved,
    using the full model's active parameter count (method-agnostic).

      MFU_hardware = (6 · N_total · tokens/s) / (n_gpus · peak_flops)

    Args:
        row: DataFrame row with model_name, gpu_model, and throughput columns

    Returns:
        float: Hardware MFU percentage or np.nan

    Reference:
        Chowdhery et al. (2022) "PaLM: Scaling Language Modeling with Pathways"
    """
    n_params = get_mfu_params(row["model_name"])
    gpu_key = normalize_gpu_model(row["gpu_model"])
    peak_flops = gpu_flops.get(gpu_key, np.nan) if gpu_key else np.nan
    total_gpus = _resolve_total_gpus(row)
    tokens_per_sec = _resolve_tokens_per_sec(row)

    if any(pd.isna(x) for x in [n_params, peak_flops, total_gpus, tokens_per_sec]):
        return np.nan
    if total_gpus == 0:
        return np.nan

    return (6 * n_params * tokens_per_sec) / (total_gpus * peak_flops) * 100


def compute_mfu_computational(row):
    """
    Computational MFU — measures efficiency of the method-specific compute:
      full      : same as MFU_hardware  (6N)
      lora      : 6 · N_lora · tokens/s  (Hu et al. 2021)
      gptq-lora : 6 · N_lora · 1.1 · tokens/s  (adds ~10% dequantisation overhead,
                                                   Frantar et al. 2022)

    A high MFU_computational with a lower MFU_hardware indicates the GPU is well
    utilised for the LoRA-specific computation but is idling on the frozen layers.

    Args:
        row: DataFrame row with model_name, method, gpu_model, and throughput columns

    Returns:
        float: Computational MFU percentage or np.nan
    """
    n_params = get_trainable_params(row)
    gpu_key = normalize_gpu_model(row["gpu_model"])
    peak_flops = gpu_flops.get(gpu_key, np.nan) if gpu_key else np.nan
    total_gpus = _resolve_total_gpus(row)
    tokens_per_sec = _resolve_tokens_per_sec(row)

    if any(pd.isna(x) for x in [n_params, peak_flops, total_gpus, tokens_per_sec]):
        return np.nan
    if total_gpus == 0:
        return np.nan

    method = str(row.get("method", "")).lower().strip()
    overhead = 1.1 if method == "gptq-lora" else 1.0

    return (6 * n_params * overhead * tokens_per_sec) / (total_gpus * peak_flops) * 100


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def nan_reason(row):
    """
    Identify why MFU calculation resulted in NaN for a given row.

    Args:
        row: DataFrame row with model_name, gpu_model, and GPU count columns

    Returns:
        str: Reason code - 'unknown_model', 'no_num_parameters', 'unknown_gpu',
             'missing_total_gpus', or 'other'
    """
    if pd.isna(get_mfu_params(row["model_name"])):
        cfg = get_model_config(row["model_name"])
        if cfg is None:
            return "unknown_model"
        return "no_num_parameters"  # model in dict but num_parameters=None
    if pd.isna(gpu_flops.get(row["gpu_model"], np.nan)):
        return "unknown_gpu"
    if pd.isna(row.get("total_gpus", np.nan)) or row.get("total_gpus", np.nan) == 0:
        return "missing_total_gpus"
    return "other"


# Made with Bob
