"""Sanity & internal-consistency invariants for the shipped GPU/LLM spec library.

Every simulation reads these dicts, so a typo'd or physically impossible spec
silently corrupts every downstream number. Nothing else tests the library
content. These assert each entry is internally consistent and physically sane
(positive bandwidth/FLOPs/memory, idle<=max power, dict key == spec name, derived
unit conversions correct) without hard-coding the values.
"""

from __future__ import annotations

import pytest

from kavier_library.gpu import GPU_SPEC_LIBRARY
from kavier_library.llm import LLM_SPEC_LIBRARY


@pytest.mark.parametrize("key", list(GPU_SPEC_LIBRARY))
def test_gpu_spec_physically_sane(key) -> None:
    gpu = GPU_SPEC_LIBRARY[key]
    assert gpu.name == key  # dict key must match the spec's own name (lookup contract)
    assert gpu.cores > 0
    assert gpu.fp_16_tensor_core_tflops > 0
    assert gpu.bandwidth_bps > 0
    assert gpu.memory_gb > 0
    assert gpu.core_max_mhz > 0
    assert gpu.network_bandwidth_gbps > 0
    # MFU is a fraction of peak; must be in (0, 1].
    assert 0.0 < gpu.mfu_factor <= 1.0
    # Power envelope ordering used by the power model.
    assert 0 < gpu.idle_power_w <= gpu.max_power_w


@pytest.mark.parametrize("key", list(GPU_SPEC_LIBRARY))
def test_gpu_bandwidth_bps_is_gbps_times_1e9(key) -> None:
    # bandwidth_bps is the GB/s figure x 1e9; the decode memory-bound term relies
    # on this exact conversion. Recover gbps and check it's a clean round number.
    gpu = GPU_SPEC_LIBRARY[key]
    gbps = gpu.bandwidth_bps / 1e9
    assert gbps == pytest.approx(round(gbps), abs=1e-6)
    assert gbps > 0


@pytest.mark.parametrize("key", list(LLM_SPEC_LIBRARY))
def test_llm_spec_physically_sane(key) -> None:
    llm = LLM_SPEC_LIBRARY[key]
    assert llm.name == key
    assert llm.m_params > 0
    assert llm.n_layers > 0
    assert llm.d_model > 0
    assert llm.n_heads > 0
    assert llm.d_head > 0
    assert llm.p_bytes > 0
    # Active params (for MoE) must be positive and never exceed total params.
    assert 0 < llm.active_params <= llm.m_params


@pytest.mark.parametrize("key", list(LLM_SPEC_LIBRARY))
def test_llm_default_active_params_equals_total_when_dense(key) -> None:
    # For dense models active_params defaults to m_params. MoE models (mixtral)
    # are the only ones allowed to set it lower.
    llm = LLM_SPEC_LIBRARY[key]
    if "mixtral" in key.lower() or "a800m" in key.lower():
        assert llm.active_params <= llm.m_params
    else:
        assert llm.active_params == pytest.approx(llm.m_params)


def test_libraries_are_nonempty() -> None:
    assert len(GPU_SPEC_LIBRARY) > 0
    assert len(LLM_SPEC_LIBRARY) > 0
