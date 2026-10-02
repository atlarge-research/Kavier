"""Predicted mean FLOPs utilization (MFU) reported by the training facade.

MFU = ``6 * N_active * tokens/s / (total_gpus * peak_flops) * 100`` (Chowdhery et al. 2022). Expected
values use the parameter count from the spec library and the peak from the NVIDIA datasheet, with the
arithmetic next to each assert.
"""

from __future__ import annotations

import math

import pytest

from kavier.sdk.training.facade import _mean_flops_utilization, performance


def test_mean_flops_utilization_hand_derived_value() -> None:
    # Llama-3-8B active_params = 8e9 (spec library); A100-80GB peak = 312 TFLOP/s (NVIDIA datasheet).
    # MFU = 6 * 8e9 * 15600 / (4 * 312e12) * 100 = 7.488e14 / 1.248e15 * 100 = 60.0 %.
    assert _mean_flops_utilization("Llama-3-8B", "A100-80GB", 15600.0, 4) == pytest.approx(60.0)


def test_mean_flops_utilization_uses_active_not_total_params_for_moe() -> None:
    # mixtral-8x7b: active_params = 1.3e10 (2 of 8 experts), total m_params = 4.7e10. MFU uses
    # active: 6 * 1.3e10 * 8000 / (2 * 312e12) * 100 = 6.24e14 / 6.24e14 * 100 = 100.0 %.
    result = _mean_flops_utilization("mixtral-8x7b-instruct-v0.1", "A100-80GB", 8000.0, 2)
    assert result == pytest.approx(100.0)
    # Total params would give 6 * 4.7e10 * 8000 / 6.24e14 * 100 = 361.5 %.
    assert result != pytest.approx(361.5)


def test_mean_flops_utilization_uses_catalog_peak_directly() -> None:
    # The denominator is the catalog fp_16_tensor_core_tflops (A100 = 312, H100-SXM = 989), the peak
    # the engine uses for throughput. For equal N, tokens and GPUs,
    # MFU_A100 / MFU_H100 = peak_H100 / peak_A100 = 989 / 312.
    a100 = _mean_flops_utilization("Llama-3-8B", "A100-80GB", 10000.0, 4)
    h100 = _mean_flops_utilization("Llama-3-8B", "H100-SXM", 10000.0, 4)
    assert a100 / h100 == pytest.approx(989 / 312)


def test_mean_flops_utilization_scales_linearly_with_throughput() -> None:
    # Structural property: with N, total_gpus and peak fixed, MFU is proportional to predicted
    # tokens/s, so doubling the throughput doubles the MFU (no additive/offset terms).
    single = _mean_flops_utilization("Llama-3-8B", "A100-80GB", 10000.0, 4)
    double = _mean_flops_utilization("Llama-3-8B", "A100-80GB", 20000.0, 4)
    assert double == pytest.approx(2.0 * single)


def test_mean_flops_utilization_is_nan_for_degenerate_denominator() -> None:
    # total_gpus = 0 (or a zero peak) zeroes the denominator; the helper returns NaN instead of raising
    # ZeroDivisionError. The facade requires num_gpus >= 1, so only a direct call reaches this.
    assert math.isnan(_mean_flops_utilization("Llama-3-8B", "A100-80GB", 15600.0, 0))


def test_performance_mfu_uses_total_gpus_across_nodes() -> None:
    # run_training puts total_gpus = num_gpus x num_nodes in the MFU denominator: 2 x 2 = 4 GPUs here;
    # num_gpus alone (2) would double the MFU. Llama-3-8B N = 8e9, A100 peak = 312 TFLOP/s.
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
    # The facade always runs calibrated: train_tokens_per_second includes the fitted method_scale and
    # gpu_compute_utilization does not. For lora/gptq-lora method_scale > 1 (~1.11 / ~1.22) and LoRA
    # optimizer and comm overhead is negligible, so realized MFU exceeds assumed; for full fine-tuning
    # (method_scale < 1) it stays below. The two are therefore not ordered in general.
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
    # gpu_compute_utilization is the assumed efficiency the engine uses to derive throughput;
    # mean_flops_utilization is the efficiency implied by that throughput after comm and optimizer time
    # lengthen the step, so realized < assumed (one GPU, so no comm or mgc effects). Both use the same
    # catalog peak, so this holds on Hopper too and the sparsity peak is not counted twice.
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
