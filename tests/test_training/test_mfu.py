"""Predicted Mean FLOPs Utilization (MFU) surfaced by the training facade.

MFU is the standard hardware definition ``6 · N_active · tokens/s / (total_gpus · peak_flops) · 100``
(Chowdhery et al. 2022), matching the supervisors' ``mfu_calculator.py`` reference. Every expected
value is hand-derived from first principles (param count from the spec library, peak from the NVIDIA
datasheet) with the arithmetic shown beside the assert — never copied from an engine run.
"""

from __future__ import annotations

import math

import pytest

from kavier.sdk.training.facade import _mean_flops_utilization, performance


def test_mean_flops_utilization_hand_derived_value() -> None:
    # Llama-3-8B active_params = 8e9 (spec library); A100-80GB peak = 312 TFLOP/s (NVIDIA datasheet).
    # MFU = 6 · 8e9 · 15600 / (4 · 312e12) · 100 = 7.488e14 / 1.248e15 · 100 = 60.0 %.
    assert _mean_flops_utilization("Llama-3-8B", "A100-80GB", 15600.0, 4) == pytest.approx(60.0)


def test_mean_flops_utilization_uses_active_not_total_params_for_moe() -> None:
    # mixtral-8x7b: active_params = 1.3e10 (2 of 8 experts), total m_params = 4.7e10. MFU must use
    # ACTIVE: 6 · 1.3e10 · 8000 / (2 · 312e12) · 100 = 6.24e14 / 6.24e14 · 100 = 100.0 %.
    result = _mean_flops_utilization("mixtral-8x7b-instruct-v0.1", "A100-80GB", 8000.0, 2)
    assert result == pytest.approx(100.0)
    # Using TOTAL params would give 6 · 4.7e10 · 8000 / 6.24e14 · 100 = 361.5 % — guard the regression.
    assert result != pytest.approx(361.5)


def test_mean_flops_utilization_uses_catalog_peak_directly() -> None:
    # The denominator is the catalog's fp_16_tensor_core_tflops as-is (A100 = 312, H100-SXM = 1979),
    # the SAME figure the engine derives throughput against, so MFU stays self-consistent. For
    # identical N/tokens/gpus, MFU_A100 / MFU_H100 = peak_H100 / peak_A100 = 1979 / 312. (No per-GPU
    # sparsity fudge is applied here: the engine and this metric must share one peak or the ratio breaks.)
    a100 = _mean_flops_utilization("Llama-3-8B", "A100-80GB", 10000.0, 4)
    h100 = _mean_flops_utilization("Llama-3-8B", "H100-SXM", 10000.0, 4)
    assert a100 / h100 == pytest.approx(1979 / 312)


def test_mean_flops_utilization_scales_linearly_with_throughput() -> None:
    # Structural property: with N, total_gpus and peak fixed, MFU is proportional to predicted
    # tokens/s, so doubling the throughput doubles the MFU (no additive/offset terms).
    single = _mean_flops_utilization("Llama-3-8B", "A100-80GB", 10000.0, 4)
    double = _mean_flops_utilization("Llama-3-8B", "A100-80GB", 20000.0, 4)
    assert double == pytest.approx(2.0 * single)


def test_mean_flops_utilization_is_nan_for_degenerate_denominator() -> None:
    # Guard, not a crash: total_gpus = 0 (or a zero peak) zeroes the denominator, so the helper returns
    # NaN rather than raising ZeroDivisionError. The facade validates num_gpus >= 1 so this is only
    # reachable by calling the helper directly, but the guard must hold.
    assert math.isnan(_mean_flops_utilization("Llama-3-8B", "A100-80GB", 15600.0, 0))


def test_performance_mfu_uses_total_gpus_across_nodes() -> None:
    # Wiring guard: run_training must feed total_gpus = num_gpus × num_nodes into the MFU denominator,
    # not num_gpus alone. With num_gpus=2, num_nodes=2 the denominator must reflect 4 GPUs; using
    # num_gpus (2) would DOUBLE the reported MFU. Pin the identity against total_gpus=4 (Llama-3-8B
    # N=8e9, A100 peak=312 TFLOP/s) — a num_gpus/total_gpus swap in run_training breaks it.
    out = performance(
        {
            "model": "Llama-3-8B",
            "gpu": "A100-80GB",
            "method": "full",
            "seq_len": 2048,
            "batch_size": 8,
            "num_gpus": 2,
            "num_nodes": 2,
            "total_tokens": 10_000_000,
        }
    ).iloc[0]
    tps = out["train_tokens_per_second"]
    assert out["mean_flops_utilization"] == pytest.approx(6 * 8e9 * tps / (4 * 312e12) * 100)


def test_realized_mfu_can_exceed_assumed_under_lora_calibration_boost() -> None:
    # The facade always runs CALIBRATED: train_tokens_per_second carries the fitted method_scale while
    # gpu_compute_utilization does not. For lora/gptq-lora method_scale > 1 (≈1.11 / ≈1.22) and the LoRA
    # optimizer/comm overhead is negligible, so realized MFU rises ABOVE assumed; for full fine-tuning
    # (method_scale < 1) it stays below. Pin BOTH directions on a calibrated pair so the ordering is a
    # tested contract — the two are deliberately NOT ordered in general (guards the docstring).
    def row(method: str) -> dict[str, object]:
        return {
            "model": "granite-3-8b",
            "gpu": "NVIDIA-A100-SXM4-80GB",
            "method": method,
            "seq_len": 2048,
            "batch_size": 8,
            "num_gpus": 1,
            "num_nodes": 1,
            "total_tokens": 10_000_000,
        }

    gptq = performance(row("gptq-lora")).iloc[0]
    full = performance(row("full")).iloc[0]
    assert gptq["mean_flops_utilization"] > gptq["gpu_compute_utilization"]
    assert full["mean_flops_utilization"] < full["gpu_compute_utilization"]


@pytest.mark.parametrize("gpu", ["A100-80GB", "L40S", "H100-SXM"])
def test_realized_mfu_is_below_assumed_compute_utilization(gpu: str) -> None:
    # gpu_compute_utilization is the *assumed* efficiency the engine uses to derive throughput;
    # mean_flops_utilization is the efficiency *implied by* the resulting throughput after
    # comm + optimizer-step overhead lengthens the step. Those overheads only reduce throughput, so
    # the realized MFU must sit strictly below the assumed one (single GPU: no comm/mgc confounders).
    # Because both use the SAME catalog peak, this self-consistency holds on every GPU, Hopper included
    # (proving the sparsity peak is not double-counted in the denominator).
    row = {
        "model": "Llama-3-8B",
        "gpu": gpu,
        "method": "full",
        "seq_len": 2048,
        "batch_size": 8,
        "num_gpus": 1,
        "num_nodes": 1,
        "total_tokens": 10_000_000,
    }
    out = performance(row).iloc[0]
    assert 0.0 < out["mean_flops_utilization"] < out["gpu_compute_utilization"]
