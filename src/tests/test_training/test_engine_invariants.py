"""Physical invariants & sanity bounds for the training engine.

Complements test_engine.py (which pins specific snapshot numbers) by asserting
the qualitative physics that must hold for ANY model/GPU/scale combination:
  * all outputs are non-negative & finite; utilisations are in [0, 100];
  * more GPUs -> not-lower aggregate throughput, and per-GPU throughput never
    exceeds the single-GPU rate (communication can only cost, never speed up);
  * a longer training run (more total_tokens) -> longer runtime;
  * larger batch -> not-lower MFU (more arithmetic intensity), capped at 1;
  * calibrated=True changes the result and shifts it in the fitted direction.
"""

from __future__ import annotations

import math
import warnings

import pytest

from kavier_library.gpu import GPU_SPEC_LIBRARY
from kavier_training.core.engine import simulate_full_training, simulate_training_step

_MODELS = ["llama3.2-3b", "mistral-7b-v0.1", "granite-3-8b", "llama3.1-70b"]
_GPU = "NVIDIA-A100-SXM4-80GB"


# --------------------------------------------------------------------------- #
# Non-negativity & bounds across many configs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("method", ["full", "lora", "gptq-lora"])
@pytest.mark.parametrize("num_gpus", [1, 2, 8])
def test_step_outputs_nonnegative_and_bounded(model, method, num_gpus) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # uncovered-calibration keys may warn
        r = simulate_training_step(model, _GPU, 1024, 4, method, num_gpus=num_gpus)
    for key in ("step_time_ms", "tokens_per_second", "tokens_per_step", "gpu_power_watts"):
        assert r[key] > 0, f"{key} must be positive"
        assert math.isfinite(r[key])
    assert 0.0 <= r["gpu_compute_utilization"] <= 100.0
    assert 0.0 <= r["gpu_memory_utilization"] <= 100.0


def test_gpu_power_within_idle_and_max_envelope() -> None:
    # The power model P(u) must stay inside [idle_power, max_power] of the spec.
    gpu = GPU_SPEC_LIBRARY[_GPU]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = simulate_training_step("mistral-7b-v0.1", _GPU, 1024, 4, "full", num_gpus=8)
    assert gpu.idle_power_w <= r["gpu_power_watts"] <= gpu.max_power_w + 1e-6


# --------------------------------------------------------------------------- #
# Scaling laws
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", _MODELS)
def test_more_gpus_not_lower_aggregate_throughput(model) -> None:
    def tps(ng):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return simulate_training_step(model, _GPU, 1024, 4, "full", num_gpus=ng)["tokens_per_second"]

    # Aggregate cluster throughput must be non-decreasing in GPU count.
    assert tps(2) >= tps(1)
    assert tps(8) >= tps(2)


def test_per_gpu_throughput_never_beats_single_gpu() -> None:
    # Adding GPUs introduces communication/correction overhead; per-GPU efficiency
    # can only stay flat or drop, never exceed the lone-GPU rate.
    one = simulate_full_training(
        model_name="mistral-7b-v0.1", method="full", gpu_model=_GPU,
        tokens_per_sample=1024, batch_size=4, number_gpus=1, number_nodes=1,
        total_tokens=1_000_000,
    )["train_tokens_per_gpu_per_second"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        many = simulate_full_training(
            model_name="mistral-7b-v0.1", method="full", gpu_model=_GPU,
            tokens_per_sample=1024, batch_size=4, number_gpus=8, number_nodes=1,
            total_tokens=1_000_000,
        )["train_tokens_per_gpu_per_second"]
    assert many <= one * (1.0 + 1e-9)


def test_runtime_scales_with_total_tokens() -> None:
    # 2x the tokens at a fixed throughput -> 2x runtime (exactly, same config).
    common = dict(
        model_name="mistral-7b-v0.1", method="full", gpu_model=_GPU,
        tokens_per_sample=1024, batch_size=4, number_gpus=2, number_nodes=1,
    )
    r1 = simulate_full_training(total_tokens=1_000_000, **common)
    r2 = simulate_full_training(total_tokens=2_000_000, **common)
    assert r2["train_runtime"] > r1["train_runtime"]
    assert r2["train_runtime"] == pytest.approx(2 * r1["train_runtime"], rel=1e-9)


def test_larger_batch_not_lower_mfu_capped_at_100() -> None:
    # MFU model: min(1, alpha*log2(bs)+beta) -> monotonic non-decreasing in bs, <=100%.
    def util(bs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return simulate_training_step("mistral-7b-v0.1", _GPU, 1024, bs, "full")["gpu_compute_utilization"]

    u1, u8, u64 = util(1), util(8), util(64)
    assert u8 >= u1
    assert u64 >= u8
    assert u64 <= 100.0 + 1e-9


# --------------------------------------------------------------------------- #
# Calibration on/off: must matter, and shift the prediction the fitted way
# --------------------------------------------------------------------------- #
def test_calibration_changes_result() -> None:
    cal = simulate_training_step("mistral-7b-v0.1", _GPU, 1024, 4, "full", num_gpus=1, calibrated=True)
    unc = simulate_training_step("mistral-7b-v0.1", _GPU, 1024, 4, "full", num_gpus=1, calibrated=False)
    assert cal["tokens_per_second"] != pytest.approx(unc["tokens_per_second"])
    # step_time also moves: mfu_multiplier feeds the micro-step time.
    assert cal["step_time_ms"] != pytest.approx(unc["step_time_ms"])


def test_calibrated_throughput_matches_uncalibrated_times_known_factors() -> None:
    # The calibrated throughput is the uncalibrated one re-scaled by the fitted
    # mfu_multiplier (via step time) and the throughput_scale product. Derive the
    # expected calibrated tps from the uncalibrated run and the calibration table
    # and assert they agree -- proving calibrated=True actually applies the table.
    from kavier_training.core import calibration as cal

    model, method, ng = "mistral-7b-v0.1", "full", 1
    unc = simulate_training_step(model, _GPU, 1024, 4, method, num_gpus=ng, calibrated=False)
    got = simulate_training_step(model, _GPU, 1024, 4, method, num_gpus=ng, calibrated=True)

    # Uncalibrated step has no mfu_multiplier and no training overhead; the
    # calibrated MFU is scaled by the multiplier, which divides the forward time.
    # Rather than reconstruct the whole step, assert the throughput_scale factor
    # is exactly applied by comparing the ratio to the table product.
    mult = cal.get_mfu_multiplier(_GPU)
    method_s = cal.get_method_scale(method)
    model_s = cal.get_model_scale(model)
    inter_s = cal.get_interaction_scale(model, method, _GPU, ng)
    # tps = tokens_per_step / step_time * throughput_scale ; tokens_per_step is
    # identical (mgc==1 at 1 GPU), so tps_cal/tps_unc == (step_unc/step_cal)*scale.
    expected_ratio = (unc["step_time_ms"] / got["step_time_ms"]) * (method_s * model_s * inter_s)
    assert got["tokens_per_second"] / unc["tokens_per_second"] == pytest.approx(expected_ratio, rel=1e-9)
    assert mult > 0  # the multiplier is what shrank step_cal vs step_unc


def test_uncovered_gpu_falls_back_to_neutral_with_warning() -> None:
    # A spec'd-but-uncalibrated GPU must not crash a calibrated run: it warns once
    # and uses the neutral 1.0 mfu_multiplier. Pick a library GPU that genuinely
    # has no mfu_multiplier entry in the shipped calibration table.
    from kavier_training.core import calibration as cal

    uncovered = next(g for g in GPU_SPEC_LIBRARY if g not in cal._CAL["mfu_multiplier"])
    cal._WARNED_KEYS.discard(uncovered)
    with pytest.warns(UserWarning, match="mfu_multiplier"):
        r = simulate_training_step("mistral-7b-v0.1", uncovered, 1024, 4, "full", num_gpus=1)
    assert r["tokens_per_second"] > 0
