"""Tests for the analytical training engine (kavier.sdk.training.core.engine).

Expected values come from hand derivations (FLOPs = 2*N*T, backward_factor, Adam 20 B/param, ring
all-reduce), a special case (the power curve at r=1), an algebraic identity (runtime = tokens /
throughput), or a structural invariant (comm overhead lowers the per-GPU rate).
"""

from __future__ import annotations

import math
import warnings
from enum import Enum

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY
from kavier.sdk.training import calibration as cal
from kavier.sdk.training.core.config import Method
from kavier.sdk.training.core.engine import simulate_full_training, simulate_training_step

_GPU = "NVIDIA-A100-SXM4-80GB"
_MODEL = "mistral-7b-v0.1"
_MODELS = ["llama3.2-3b", "mistral-7b-v0.1", "granite-3-8b", "llama3.1-70b"]

# alpha, beta copied from calibration.json's mfu_batch_scale, so the arithmetic below does not use
# the engine's curve.
_ALPHA, _BETA = 0.0341, 0.8147
# Adam optimizer traffic per trainable parameter [bytes].
_OPT_BYTES = 20


def _step(**kw):
    """Run simulate_training_step with default arguments and calibration warnings suppressed."""
    d = dict(model_name=_MODEL, gpu_model=_GPU, tokens_per_sample=1024, batch_size=4, method="full")
    d.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return simulate_training_step(**d)


# --- input validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"grad_accum_steps": 0}, "grad_accum_steps must be >= 1"),
        ({"backward_factor": 0.0}, "backward_factor must be > 0"),
        ({"tokens_per_sample": 0}, "tokens_per_sample must be >= 1"),
        ({"num_gpus": 0}, "num_gpus must be >= 1"),
    ],
)
def test_invalid_inputs_raise(kw, msg) -> None:
    # Without the guard these inputs return a number or divide by zero.
    with pytest.raises(ValueError, match=msg):
        _step(**kw)


# --- method names -------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["LoRA", "Full", "bogus", ""])
def test_unknown_method_raises_and_names_the_valid_methods(method) -> None:
    # An unknown name used to get full fine-tuning physics with no error.
    with pytest.raises(ValueError, match=r"unknown method .*full.*gptq-lora.*lora.*qlora"):
        _step(method=method)


@pytest.mark.parametrize("calibrated", [False, True])
def test_method_enum_member_gives_the_same_step_as_its_string(calibrated) -> None:
    assert _step(method=Method.LORA, calibrated=calibrated) == _step(method="lora", calibrated=calibrated)


def test_other_str_enum_method_gives_the_same_step_as_its_string() -> None:
    # f"{member}" of a (str, Enum) renders "Adapter.LORA", which missed the interaction_scale key.
    # num_gpus=2 is an interaction_scale cell of the shipped table.
    class Adapter(str, Enum):
        LORA = "lora"

    kw = dict(model_name="granite-3.1-2b", num_gpus=2, calibrated=True)
    assert "granite-3.1-2b|lora|NVIDIA-A100-SXM4-80GB|2" in cal._active_calibration()["interaction_scale"]
    assert _step(method=Adapter.LORA, **kw) == _step(method="lora", **kw)


def test_full_training_reports_the_method_as_a_plain_string() -> None:
    r = _full(method=Method.GPTQ_LORA)
    assert r["method"] == "gptq-lora"
    assert type(r["method"]) is str


@pytest.mark.parametrize("method", ["qlora", "alora"])
def test_adapter_methods_train_the_same_parameters_as_lora(method) -> None:
    # Uncalibrated, method only changes the trainable-parameter count, so these adapter methods and lora
    # give the same step and differ from full.
    lora = _step(method="lora", calibrated=False, num_gpus=2)
    assert _step(method=method, calibrated=False, num_gpus=2) == lora
    assert _step(method="full", calibrated=False, num_gpus=2) != lora


@pytest.mark.parametrize("method", ["qlora", "alora"])
def test_adapter_methods_without_a_fit_fall_back_to_a_neutral_method_scale(method) -> None:
    cal._WARNED_KEYS.discard(method)
    with pytest.warns(UserWarning, match=f"no method_scale entry for '{method}'"):
        simulate_training_step(_MODEL, _GPU, 1024, 4, method)


def test_method_present_only_in_the_live_calibration_is_accepted() -> None:
    # calibrate() seeds method_scale for methods found in the data (e.g. fsdp) and predicts with them.
    saved = cal._active_calibration()
    cal._CAL = {**saved, "method_scale": {**saved["method_scale"], "fsdp": 1.0}}
    try:
        r = _step(method="fsdp")
    finally:
        cal._CAL = saved
    assert r["tokens_per_second"] > 0
    with pytest.raises(ValueError, match="unknown method 'fsdp'"):
        _step(method="fsdp")


# --- step time from first principles ------------------------------------------------------------


def test_uncalibrated_step_time_matches_first_principles() -> None:
    # mistral-7b (active=7e9), A100 (312 TFLOP/s fp16, 2.039e12 B/s, mfu_factor=0.4513), bs=4, T=1024.
    # forward FLOPs = 2*N*T = 2*7e9*4096 = 5.7344e13.
    # mfu (uncalibrated) = 0.4513 * min(1, 0.0341*log2(4)+0.8147) = 0.4513*0.8829 = 0.39845277.
    # forward_time = 5.7344e13 / (312e12*0.39845277) = 0.4612714 s.
    # step = forward*(1+backward_factor) + optimizer = 3*forward + 7e9*20/2.039e12
    #      = 1.3838142 + 0.0686611 = 1.4524753 s -> 1452.4753 ms.
    r = _step(calibrated=False)
    assert r["step_time_ms"] == pytest.approx(1452.4753, rel=1e-5)


def test_backward_factor_adds_one_forward_time_per_unit() -> None:
    # At num_gpus=1 step = (1+bf)*forward + optimizer, so raising bf from 2 to 5 adds 3*forward.
    # forward is recovered from the returned MFU: forward = 2NT/(peak*mfu).
    r2 = _step(calibrated=False, backward_factor=2.0)
    r5 = _step(calibrated=False, backward_factor=5.0)
    mfu = r2["gpu_compute_utilization"] / 100.0
    forward_ms = (2 * 7e9 * 4 * 1024) / (312e12 * mfu) * 1000.0
    assert r5["step_time_ms"] - r2["step_time_ms"] == pytest.approx(3 * forward_ms, rel=1e-9)


def test_lora_cuts_optimizer_time_by_the_trainable_param_delta() -> None:
    # Only the optimizer term depends on trainable-param count. full trains all m_params (7e9);
    # LoRA trains 2*rank*d_model*target_modules*n_layers = 2*8*4096*4*32 = 8_388_608 params.
    # So step_full - step_lora = (7e9 - 8_388_608) * 20 / 2.039e12 s = 68.5788 ms.
    lora_trainable = 2 * 8 * 4096 * 4 * 32
    full = _step(calibrated=False, method="full")
    lora = _step(calibrated=False, method="lora")
    expected_ms = (7e9 - lora_trainable) * _OPT_BYTES / 2.039e12 * 1000.0
    assert full["step_time_ms"] - lora["step_time_ms"] == pytest.approx(expected_ms, rel=1e-9)


# --- MFU / compute utilization ------------------------------------------------------------------


def test_compute_utilization_is_mfu_factor_times_batch_curve() -> None:
    # Below the cap, util% = 100 * mfu_factor * (alpha*log2(bs) + beta). bs=4 -> log2=2.
    gpu = GPU_SPEC_LIBRARY[_GPU]
    expected = 100.0 * gpu.mfu_factor * (_ALPHA * math.log2(4) + _BETA)  # 100*0.4513*0.8829 = 39.845
    assert _step(calibrated=False)["gpu_compute_utilization"] == pytest.approx(expected, rel=1e-9)


def test_batch_scale_caps_mfu_at_the_gpu_factor() -> None:
    # alpha*log2(64)+beta = 0.0341*6+0.8147 = 1.0193 > 1, so batch_scale clamps to 1 and
    # util is mfu_factor*100 (45.13% for the A100) for any larger batch.
    gpu = GPU_SPEC_LIBRARY[_GPU]
    r = _step(calibrated=False, batch_size=64)
    assert r["gpu_compute_utilization"] == pytest.approx(gpu.mfu_factor * 100.0, rel=1e-9)


@given(bs_lo=st.integers(min_value=1, max_value=200), bump=st.integers(min_value=0, max_value=300))
def test_compute_utilization_monotonic_nondecreasing_in_batch(bs_lo: int, bump: int) -> None:
    # batch_scale = min(1, alpha*log2(bs)+beta) rises with bs then plateaus => util never decreases.
    lo = _step(calibrated=False, batch_size=bs_lo)["gpu_compute_utilization"]
    hi = _step(calibrated=False, batch_size=bs_lo + bump)["gpu_compute_utilization"]
    assert hi >= lo - 1e-9


def test_calibration_scales_compute_util_by_the_gpu_mfu_multiplier() -> None:
    # Calibrated and raw MFU differ only by the per-GPU mfu_multiplier (the batch curve is shared),
    # so util_cal / util_unc equals that multiplier.
    unc = _step(calibrated=False)["gpu_compute_utilization"]
    got = _step(calibrated=True)["gpu_compute_utilization"]
    assert got / unc == pytest.approx(cal.get_mfu_multiplier(_GPU), rel=1e-9)


# --- power model --------------------------------------------------------------------------------


@pytest.mark.parametrize("gpu_name", sorted(GPU_SPEC_LIBRARY))
def test_power_is_linear_ramp_from_idle_to_max(gpu_name) -> None:
    # Every shipped GPU has r = mse_calib_factor = 1, so idle+(max-idle)*(2u-u^r) reduces to
    # idle+(max-idle)*u with u = max(compute_util, mem_util); 0 <= u <= 1 keeps power in [idle, max].
    gpu = GPU_SPEC_LIBRARY[gpu_name]
    assert gpu.mse_calib_factor == 1.0  # the reduction above needs r = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = simulate_training_step(_MODEL, gpu_name, 1024, 4, "full", num_gpus=8, calibrated=False)
    u = max(r["gpu_compute_utilization"], r["gpu_memory_utilization"]) / 100.0
    expected = gpu.idle_power_w + (gpu.max_power_w - gpu.idle_power_w) * u
    assert r["gpu_power_watts"] == pytest.approx(expected, rel=1e-9)
    assert gpu.idle_power_w <= r["gpu_power_watts"] <= gpu.max_power_w + 1e-9


# --- tokens per step (data parallel) ------------------------------------------------------------


@pytest.mark.parametrize("num_gpus", [1, 2, 8])
@pytest.mark.parametrize("grad_accum", [1, 3])
def test_tokens_per_step_is_dataparallel_product_uncalibrated(num_gpus, grad_accum) -> None:
    # Uncalibrated multi_gpu_correction is 1, so tokens/step = grad_accum * batch * seq * num_gpus
    # (4*1024 tokens per GPU micro-step).
    r = _step(calibrated=False, num_gpus=num_gpus, grad_accum_steps=grad_accum)
    assert r["tokens_per_step"] == pytest.approx(grad_accum * 4 * 1024 * num_gpus)


def test_calibrated_tokens_per_step_is_the_dataparallel_product() -> None:
    # A step processes grad_accum * batch * seq * num_gpus tokens with or without calibration;
    # multi_gpu_correction(8) lowers the step rate instead.
    assert _step(calibrated=True, num_gpus=8)["tokens_per_step"] == 4 * 1024 * 8


def test_calibrated_steps_per_second_includes_the_multi_gpu_correction() -> None:
    # steps/s = throughput_scale / (mgc(8) * step_time). Dividing tokens_per_step by mgc instead
    # left mgc out of steps/s.
    step = _step(calibrated=True, num_gpus=8)
    r = _full(number_gpus=8)
    scale = (
        cal.get_method_scale("full") * cal.get_model_scale(_MODEL) * cal.get_interaction_scale(_MODEL, "full", _GPU, 8)
    )
    expected = scale / (cal.get_multi_gpu_correction(8) * step["step_time_ms"] / 1000.0)
    assert r["train_steps_per_second"] == pytest.approx(expected, rel=1e-9)


# --- multi-GPU scaling invariants (structural, uncalibrated) ------------------------------------


@pytest.mark.parametrize("model", _MODELS)
def test_aggregate_throughput_strictly_increases_with_gpus(model) -> None:
    # Data-parallel tokens grow linearly with GPU count while step time gains a sublinear
    # all-reduce term, so aggregate tokens/s strictly increases.
    def tps(ng):
        return _step(model_name=model, calibrated=False, num_gpus=ng)["tokens_per_second"]

    assert tps(1) < tps(2) < tps(8)


@pytest.mark.parametrize("model", _MODELS)
def test_per_gpu_throughput_strictly_decreases_with_gpus(model) -> None:
    # Communication overhead makes the per-GPU rate (aggregate/num_gpus) fall as GPUs are added.
    def per_gpu(ng):
        return _step(model_name=model, calibrated=False, num_gpus=ng)["tokens_per_second"] / ng

    assert per_gpu(1) > per_gpu(2) > per_gpu(8)


# --- calibration application (cross-check) -------------------------------------------------------


def test_calibrated_throughput_reconstructs_from_documented_factors() -> None:
    # At 1 GPU, calibrated=True multiplies throughput by method_scale*model_scale*interaction_scale
    # and shortens step time through mfu_multiplier. tokens/step is unchanged (mgc=1 at 1 GPU), so
    #   tps_cal/tps_unc == (step_unc/step_cal) * (method*model*interaction).
    # The ratio is rebuilt from the calibration accessors; a dropped scale factor breaks it.
    unc = _step(calibrated=False, num_gpus=1)
    got = _step(calibrated=True, num_gpus=1)
    scale = (
        cal.get_method_scale("full") * cal.get_model_scale(_MODEL) * cal.get_interaction_scale(_MODEL, "full", _GPU, 1)
    )
    expected_ratio = (unc["step_time_ms"] / got["step_time_ms"]) * scale
    assert got["tokens_per_second"] / unc["tokens_per_second"] == pytest.approx(expected_ratio, rel=1e-9)
    # mfu_multiplier changes the step time, so the two step times differ.
    assert got["step_time_ms"] != pytest.approx(unc["step_time_ms"])


# --- full-training extrapolation ----------------------------------------------------------------


def _full(**kw):
    d = dict(
        model_name=_MODEL,
        method="full",
        gpu_model=_GPU,
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=1,
        number_nodes=1,
    )
    d.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return simulate_full_training(**d)


def test_full_training_reports_are_exact_functions_of_throughput() -> None:
    # The per-GPU, per-sample and per-step rates and the runtime are exact functions of the
    # throughput and the job geometry.
    r = _full(number_gpus=8, number_nodes=1, total_tokens=1_000_000)
    tps = r["train_tokens_per_second"]
    assert r["train_tokens_per_gpu_per_second"] == pytest.approx(tps / 8, rel=1e-12)
    assert r["train_samples_per_second"] == pytest.approx(tps / 1024, rel=1e-12)
    assert r["train_runtime"] == pytest.approx(1_000_000 / tps, rel=1e-12)
    # steps/s = tps / tokens_per_step, with tokens_per_step from a matching single step
    step = _step(calibrated=True, num_gpus=8)
    assert r["train_steps_per_second"] == pytest.approx(tps / step["tokens_per_step"], rel=1e-9)


def test_runtime_is_linear_in_total_tokens_and_zero_when_unset() -> None:
    # runtime = total_tokens / throughput at fixed config: doubling tokens exactly doubles runtime.
    r1 = _full(total_tokens=1_000_000)
    r2 = _full(total_tokens=2_000_000)
    assert r2["train_runtime"] == pytest.approx(2 * r1["train_runtime"], rel=1e-12)
    # no job size -> runtime 0.0
    assert _full()["train_runtime"] == 0.0


def test_epochs_times_dataset_tokens_equals_total_tokens() -> None:
    # total_tokens := round(epochs * dataset_tokens); 2 * 500_000 == a direct 1_000_000 request.
    via_epochs = _full(epochs=2, dataset_tokens=500_000)
    direct = _full(total_tokens=1_000_000)
    assert via_epochs["total_tokens"] == 1_000_000
    assert via_epochs["train_runtime"] == pytest.approx(direct["train_runtime"], rel=1e-12)


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"epochs": 2}, "epochs and dataset_tokens together"),
        ({"epochs": -1, "dataset_tokens": 100}, "must be non-negative"),
    ],
)
def test_bad_epochs_dataset_pairs_raise(kw, msg) -> None:
    # only one of (epochs, dataset_tokens) given, or a negative value
    with pytest.raises(ValueError, match=msg):
        _full(**kw)


def test_negative_total_tokens_raises() -> None:
    # Used to return a negative runtime.
    with pytest.raises(ValueError, match="total_tokens must be non-negative"):
        _full(total_tokens=-1_000_000)


@pytest.mark.parametrize(
    "gpus, nodes, msg",
    [
        (-2, -1, "number_gpus must be >= 1, got -2"),  # the product 2 used to pass
        (0, 1, "number_gpus must be >= 1, got 0"),
        (2, -1, "number_nodes must be >= 1, got -1"),
        (1, 0, "number_nodes must be >= 1, got 0"),
    ],
)
def test_full_training_checks_gpu_and_node_counts_separately(gpus, nodes, msg) -> None:
    with pytest.raises(ValueError, match=msg):
        _full(number_gpus=gpus, number_nodes=nodes)


# --- calibration fallback -----------------------------------------------------------------------


def test_uncovered_gpu_falls_back_to_neutral_with_warning() -> None:
    # A calibrated run on a GPU missing from the mfu_multiplier table warns once, uses 1.0 and
    # still returns a positive throughput.
    covered = cal._active_calibration()["mfu_multiplier"]
    uncovered = next(g for g in GPU_SPEC_LIBRARY if g not in covered)
    cal._WARNED_KEYS.discard(uncovered)
    with pytest.warns(UserWarning, match="mfu_multiplier"):
        r = simulate_training_step(_MODEL, uncovered, 1024, 4, "full", num_gpus=1)
    assert r["tokens_per_second"] > 0
