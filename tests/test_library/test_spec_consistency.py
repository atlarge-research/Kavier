"""Consistency checks for the shipped GPU/LLM spec library and the two spec constructors.

The spec objects do no validation, so this file is the validation layer: physical invariants of every
catalog entry (positive fields, ``idle <= max`` power, ``key == name``, ``0 < active_params <= m_params``)
and the constructors' derived values (GB/s -> bytes/s, power-envelope defaults, ``active_params``
default), checked with values that do not come from the catalog.
"""

from __future__ import annotations

import pytest

from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY
from kavier.sdk.library.llm import LLM_SPEC_LIBRARY
from kavier.sdk.library.specs.GPUSpec import GPUSpec
from kavier.sdk.library.specs.LLMSpec import LLMSpec


def _make_gpu(**overrides) -> GPUSpec:
    """Return a GPUSpec with fixed required arguments and the given overrides."""
    kwargs = dict(
        gpu_name="TEST",
        memory_bandwidth_gbps=1000,
        fp_16_tensor_core_tflops=312,
        gpu_cores=6912,
        memory_gb=80,
        gpu_core_max_mhz=1410,
        base_power_w=200,
    )
    kwargs.update(overrides)
    return GPUSpec(**kwargs)


# --------------------------------------------------------------------------------------
# Constructor behavior: derived values (no catalog data)
# --------------------------------------------------------------------------------------


def test_gpuspec_converts_gbps_to_bytes_per_second() -> None:
    # 1 GB/s = 1e9 bytes/s, so 1000 GB/s -> 1000 * 1e9 = 1e12 bytes/s.
    gpu = _make_gpu(memory_bandwidth_gbps=1000)
    assert gpu.bandwidth_bps == 1e12


def test_gpuspec_power_envelope_defaults_to_quarter_base_and_base() -> None:
    # Default: idle = base * 0.25, max = base (TDP); base=200 -> idle=50, max=200.
    gpu = _make_gpu(base_power_w=200, idle_power_w=None, max_power_w=None)
    assert gpu.idle_power_w == 50.0
    assert gpu.max_power_w == 200.0


def test_gpuspec_explicit_power_values_override_defaults() -> None:
    # Given idle/max values are stored as is; base=200 would default idle to 50.
    gpu = _make_gpu(base_power_w=200, idle_power_w=30, max_power_w=150)
    assert gpu.idle_power_w == 30
    assert gpu.max_power_w == 150


def test_llmspec_active_params_defaults_to_total_when_omitted() -> None:
    # Dense default: active_params falls back to m_params.
    llm = LLMSpec(llm_name="T", n_layers=1, d_model=8, p_bytes=2, m_params=8e9, n_heads=2, d_head=4)
    assert llm.active_params == 8e9


def test_llmspec_active_params_preserved_when_supplied() -> None:
    # MoE: an explicit active_params below the total is kept.
    llm = LLMSpec(
        llm_name="T",
        n_layers=1,
        d_model=8,
        p_bytes=2,
        m_params=47e9,
        n_heads=2,
        d_head=4,
        active_params=13e9,
    )
    assert llm.active_params == 13e9


# --------------------------------------------------------------------------------------
# Catalog invariants for every shipped entry
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(GPU_SPEC_LIBRARY))
def test_gpu_catalog_entry_is_physically_sane(key) -> None:
    gpu = GPU_SPEC_LIBRARY[key]
    # get_gpu relies on the dict key equal to the spec name
    assert gpu.name == key
    assert gpu.cores > 0
    assert gpu.fp_16_tensor_core_tflops > 0
    assert gpu.bandwidth_bps > 0
    assert gpu.memory_gb > 0
    assert gpu.core_max_mhz > 0
    assert gpu.network_bandwidth_gbps > 0
    # MFU is a fraction of peak FLOPs, in (0, 1]
    assert 0.0 < gpu.mfu_factor <= 1.0
    # Power envelope ordering assumed by the mse_power model: 0 < idle <= max.
    assert 0 < gpu.idle_power_w <= gpu.max_power_w


@pytest.mark.parametrize("key", list(LLM_SPEC_LIBRARY))
def test_llm_catalog_entry_is_physically_sane(key) -> None:
    llm = LLM_SPEC_LIBRARY[key]
    # get_llm relies on the dict key equal to the spec name
    assert llm.name == key
    assert llm.m_params > 0
    assert llm.n_layers > 0
    assert llm.d_model > 0
    assert llm.n_heads > 0
    assert llm.d_head > 0
    assert llm.p_bytes > 0
    assert 0 < llm.active_params <= llm.m_params


def test_moe_models_expose_fewer_active_than_total_params() -> None:
    # MoE models have architecture-derived active_params below the total.
    # mixtral-8x7b: 2 of 8 experts active, ~13B active of 47B total.
    # granite-3.1-3b-a800m: "a800m" means 800M active of 3.3B total.
    mixtral = LLM_SPEC_LIBRARY["mixtral-8x7b-instruct-v0.1"]
    granite_moe = LLM_SPEC_LIBRARY["granite-3.1-3b-a800m-instruct"]
    assert mixtral.active_params == 13e9
    assert mixtral.active_params < mixtral.m_params
    assert granite_moe.active_params == 800e6
    assert granite_moe.active_params < granite_moe.m_params
    # Granite 4.0 hybrid MoE, from IBM's published figures: H Small 9B active of 32B, H Tiny 1B of 7B.
    h_small = LLM_SPEC_LIBRARY["granite-4.0-h-small"]
    h_tiny = LLM_SPEC_LIBRARY["granite-4.0-h-tiny"]
    assert h_small.active_params == 9e9 and h_small.active_params < h_small.m_params
    assert h_tiny.active_params == 1e9 and h_tiny.active_params < h_tiny.m_params


def test_dense_models_have_active_params_equal_to_total() -> None:
    # Every non-MoE entry has active_params equal to m_params.
    moe_keys = {
        "mixtral-8x7b-instruct-v0.1",
        "granite-3.1-3b-a800m-instruct",
        "granite-4.0-h-small",  # 9B active of 32B (IBM Granite 4.0 H Small)
        "granite-4.0-h-tiny",  # 1B active of 7B (IBM Granite 4.0 H Tiny)
    }
    for key, llm in LLM_SPEC_LIBRARY.items():
        if key in moe_keys:
            continue
        assert llm.active_params == llm.m_params, key


def test_catalog_contains_documented_shipped_keys() -> None:
    # A non-empty catalog keeps the parametrized invariants meaningful; these exact-match keys are
    # documented as shipped for both engines (case-sensitive, no aliases).
    assert {"A100-80GB", "NVIDIA-A100-SXM4-80GB"} <= set(GPU_SPEC_LIBRARY)
    assert {"Llama-3-8B", "granite-3-8b"} <= set(LLM_SPEC_LIBRARY)
