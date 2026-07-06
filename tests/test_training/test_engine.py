"""Rebuilt to the writing-tests standard: every assertion has an independent oracle
(hand-derived physics, an invariant/bound, a scaling law, or an internal cross-check), never a
snapshot of the engine's own output. Derivations are in comments next to each assert.

Reference constants used for hand-derivation (from the spec library, the oracle's source):
  mistral-7b-v0.1 : m_params = active_params = 7e9, d_model = 4096, n_layers = 32
  NVIDIA-A100-SXM4-80GB : fp16 tensor TFLOPs = 312, mfu_factor = 0.4513, bandwidth = 2039 GB/s,
                          network = 4800 GB/s, idle_power = 75 W, max_power = 400 W, mse_calib r = 1.0
Engine model (kavier.sdk.training.core.engine):
  flops       = 2 * active_params * batch * tokens
  forward     = flops / (tflops*1e12 * mfu) + overhead
  micro_step  = (1 + backward_factor) * forward
  optimizer   = trainable_params * 20 / bandwidth_bps
  step_time   = grad_accum*micro_step + optimizer + comm
  mfu         = mfu_factor * mfu_multiplier(gpu) * min(1, alpha*log2(batch)+beta)
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from kavier.sdk.io.training_opendc import build_training_opendc_frames, export_training_opendc
from kavier.sdk.library.lookup import get_gpu
from kavier.sdk.training.core.engine import simulate_full_training, simulate_training_step

# Vary the real behavior (model x gpu x method), not just a number: covers dense/MoE names,
# three GPUs, and full/lora/gptq-lora branches.
TRAINING_CONFIGS = [
    {"model_name": "mistral-7b-v0.1", "gpu_model": "NVIDIA-A100-SXM4-80GB", "method": "full"},
    {"model_name": "mistral-7b-v0.1", "gpu_model": "NVIDIA-A100-80GB-PCIe", "method": "lora"},
    {"model_name": "granite-3.3-8b", "gpu_model": "NVIDIA-A100-SXM4-80GB", "method": "full"},
    {"model_name": "granite-3-8b", "gpu_model": "NVIDIA-H100-PCIe", "method": "lora"},
    {"model_name": "llama3.2-3b", "gpu_model": "NVIDIA-A100-SXM4-80GB", "method": "gptq-lora"},
]

_IDS = [f"{c['model_name']}_{c['method']}" for c in TRAINING_CONFIGS]


@pytest.mark.parametrize("cfg", TRAINING_CONFIGS, ids=_IDS)
def test_step_metrics_obey_physical_bounds_over_catalog(cfg):
    # Invariant over the whole model x gpu x method catalog. The load-bearing oracle is the power
    # law: every shipped GPU has r = mse_calib_factor = 1.0, so mse_power = idle + (max-idle)*u
    # collapses to a LINEAR ramp in u = max(compute_util, mem_util) -- a DIFFERENT form than the
    # engine's 2u - u**r. That pins power exactly AND bounds it to [idle, max].
    gpu = get_gpu(cfg["gpu_model"])
    r = simulate_training_step(
        model_name=cfg["model_name"],
        gpu_model=cfg["gpu_model"],
        tokens_per_sample=1024,
        batch_size=4,
        method=cfg["method"],
        num_gpus=1,
    )
    assert r["step_time_ms"] > 0
    assert r["tokens_per_second"] > 0
    assert 0.0 <= r["gpu_memory_utilization"] <= 100.0  # util is a fraction of bandwidth, in [0,100]
    u = max(r["gpu_compute_utilization"], r["gpu_memory_utilization"]) / 100.0
    assert r["gpu_power_watts"] == pytest.approx(gpu.idle_power_w + (gpu.max_power_w - gpu.idle_power_w) * u, rel=1e-9)


def test_raw_physics_step_time_and_throughput_hand_derived():
    # Fully independent first-principles oracle with calibrated=False and batch_size=64 so that
    # batch_scale = min(1, 0.0341*log2(64)+0.8147) = min(1, 1.019) CLAMPS to 1.0 -> mfu = mfu_factor
    # exactly (0.4513), independent of the fitted alpha/beta.
    #   flops     = 2 * 7e9 * (64*1024) = 9.17504e14
    #   achieved  = 312e12 * 0.4513     = 1.408056e14 FLOP/s
    #   forward   = 9.17504e14 / 1.408056e14 = 6.516096 s   (overhead = 0 uncalibrated)
    #   step      = 3*forward + 7e9*20/2039e9 = 19.548287 + 0.068661 = 19.616948 s
    flops = 2 * 7e9 * (64 * 1024)
    achieved = 312e12 * 0.4513
    step_s = 3 * (flops / achieved) + 7e9 * 20 / 2039e9
    tokens_per_step = 64 * 1024  # data-parallel tokens for 1 GPU, mgc = 1 uncalibrated

    r = simulate_training_step(
        "mistral-7b-v0.1", "NVIDIA-A100-SXM4-80GB", 1024, 64, "full", num_gpus=1, calibrated=False
    )
    assert r["step_time_ms"] == pytest.approx(step_s * 1000, rel=1e-9)  # ~19616.97 ms
    assert r["tokens_per_second"] == pytest.approx(tokens_per_step / step_s, rel=1e-9)  # ~3340.78
    assert r["gpu_compute_utilization"] == pytest.approx(45.13, rel=1e-9)  # mfu*100 = 0.4513*100


def test_lora_faster_than_full_by_optimizer_delta():
    # Compute (forward/backward) FLOPs are method-independent (they use active_params), so on a
    # single GPU the ONLY step-time difference between full and lora is the optimizer traffic over
    # trainable params. Hand-derived delta:
    #   full trainable = 7e9 ; lora trainable = 2*rank*d_model*target_modules*n_layers
    #                    = 2*8*4096*4*32 = 8_388_608
    #   delta_step_ms  = (7e9 - 8_388_608) * 20 / 2039e9 * 1000 = 68.5788 ms
    lora_trainable = 2 * 8 * 4096 * 4 * 32
    expected_delta_ms = (7e9 - lora_trainable) * 20 / 2039e9 * 1000

    full = simulate_training_step("mistral-7b-v0.1", "NVIDIA-A100-SXM4-80GB", 1024, 4, "full", num_gpus=1)
    lora = simulate_training_step("mistral-7b-v0.1", "NVIDIA-A100-SXM4-80GB", 1024, 4, "lora", num_gpus=1)
    assert full["step_time_ms"] - lora["step_time_ms"] == pytest.approx(expected_delta_ms, rel=1e-9)


def test_multi_gpu_strong_scaling_is_sublinear():
    # Scaling law (calibrated=False to isolate physics from the fitted multi-gpu correction):
    # data-parallel N GPUs process N x the tokens/step but only add sublinear all-reduce comm, so
    # 1 < tps(4)/tps(1) < 4. The upper bound catches a "4 GPUs slower/equal" regression that a bare
    # `r4 > r1*0.5` would pass; the lower bound catches "more GPUs don't help".
    t1 = simulate_training_step(
        "mistral-7b-v0.1", "NVIDIA-A100-SXM4-80GB", 1024, 4, "full", num_gpus=1, calibrated=False
    )["tokens_per_second"]
    t4 = simulate_training_step(
        "mistral-7b-v0.1", "NVIDIA-A100-SXM4-80GB", 1024, 4, "full", num_gpus=4, calibrated=False
    )["tokens_per_second"]
    assert 1.0 < t4 / t1 < 4.0


def test_grad_accum_amortizes_comm():
    # G>1 spreads the fixed optimizer+comm cost over G micro-steps, so throughput rises; the gain is
    # larger where comm is larger (8 GPUs have all-reduce, 1 GPU has none). Property, not a snapshot.
    def tps(ng, g):
        return simulate_training_step(
            "mistral-7b-v0.1", "NVIDIA-A100-SXM4-80GB", 1024, 4, "full", num_gpus=ng, grad_accum_steps=g
        )["tokens_per_second"]

    ratio_1gpu = tps(1, 4) / tps(1, 1)
    ratio_8gpu = tps(8, 4) / tps(8, 1)
    assert ratio_8gpu > 1.0
    assert ratio_8gpu > ratio_1gpu  # comm amortization is bigger where comm is bigger


def test_backward_factor_lowers_throughput():
    # micro_step = (1+backward_factor)*forward, so a heavier backward pass raises step time and
    # lowers throughput. Monotonic invariant.
    def tps(bf):
        return simulate_training_step(
            "mistral-7b-v0.1", "NVIDIA-A100-SXM4-80GB", 1024, 4, "full", num_gpus=1, backward_factor=bf
        )["tokens_per_second"]

    assert tps(3.0) < tps(2.0)


def test_memory_util_uses_gb_not_gib():
    # Regression guard for the GiB->GB fix. The engine's bandwidth denominator is bandwidth_bps/1e9
    # (GB/s), so the traffic numerator must also divide by 1e9. Hand-derived:
    #   param_traffic  = 7e9 params * 2 bytes * 5 memory passes = 7e10 bytes
    #   activation     = batch(4)*seq(1024)*d_model(4096)*2 bytes
    #   util%          = min(1, (traffic/1e9 / step_s) / 2039 GB/s) * 100
    # The OLD bug divided the numerator by 2**30 (GiB), understating util by ~7%.
    r = simulate_training_step("mistral-7b-v0.1", "NVIDIA-A100-SXM4-80GB", 1024, 4, "full", num_gpus=1)
    step_s = r["step_time_ms"] / 1000.0
    traffic_bytes = 7e9 * 2 * 5 + 4 * 1024 * 4096 * 2

    correct_util = min(1.0, (traffic_bytes / 1e9 / step_s) / 2039.0) * 100.0
    gib_bug_util = min(1.0, (traffic_bytes / (1 << 30) / step_s) / 2039.0) * 100.0
    assert r["gpu_memory_utilization"] == pytest.approx(correct_util, rel=1e-9)
    assert r["gpu_memory_utilization"] != pytest.approx(gib_bug_util)  # would regress if GiB returns


def test_unknown_model_raises_keyerror():
    # get_llm raises UnknownSpecError (a KeyError subclass) for an uncatalogued name.
    with pytest.raises(KeyError):
        simulate_training_step("nonexistent-model", "NVIDIA-A100-SXM4-80GB", 1024, 4, "full")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size must be >= 1"),  # guards math.log2(batch) in the MFU model
        ({"batch_size": -1}, "batch_size must be >= 1"),
        ({"grad_accum_steps": 0}, "grad_accum_steps must be >= 1"),
        ({"backward_factor": 0.0}, "backward_factor must be > 0"),
        ({"tokens_per_sample": 0}, "tokens_per_sample must be >= 1"),
        ({"num_gpus": 0}, "num_gpus must be >= 1"),
    ],
)
def test_invalid_arguments_raise_valueerror(kwargs, message):
    call = {
        "model_name": "mistral-7b-v0.1",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 4,
        "method": "full",
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=message):
        simulate_training_step(**call)


def test_runtime_zero_without_total_tokens():
    # Job-size branch: with no total_tokens (and no epochs/dataset), runtime is 0.0 -- an unknown-
    # length job, not None and not an extrapolation. Throughput is still reported.
    r = simulate_full_training(
        model_name="mistral-7b-v0.1",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=1,
        number_nodes=1,
    )
    assert r["train_tokens_per_second"] > 0
    assert r["train_runtime"] == 0.0


def test_full_training_derived_fields_are_consistent_single_node():
    # Internal cross-checks (no pinned magic numbers): every derived field is a stated function of
    # train_tokens_per_second, and total_gpus = number_gpus*number_nodes = 1. These catch the field-
    # wiring bugs the file's history called out (/total_gpus and steps/s).
    r = simulate_full_training(
        model_name="mistral-7b-v0.1",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=1,
        number_nodes=1,
        total_tokens=1_000_000,
    )
    step = simulate_training_step("mistral-7b-v0.1", "NVIDIA-A100-SXM4-80GB", 1024, 4, "full", num_gpus=1)
    tps = r["train_tokens_per_second"]
    assert r["number_gpus"] == 1
    assert r["train_tokens_per_gpu_per_second"] == pytest.approx(tps / 1, rel=1e-12)  # 1 total GPU
    assert r["train_samples_per_second"] == pytest.approx(tps / 1024, rel=1e-12)  # tokens_per_sample
    assert r["train_steps_per_second"] == pytest.approx(tps / step["tokens_per_step"], rel=1e-12)
    assert r["train_runtime"] == pytest.approx(1_000_000 / tps, rel=1e-12)


def test_full_training_multi_node_divides_by_total_gpus():
    # Multi-node: per-gpu throughput divides by TOTAL gpus (gpus/node * nodes = 8*2 = 16), not just
    # gpus/node; steps/s stays consistent with the engine's own tokens_per_step.
    r = simulate_full_training(
        model_name="mistral-7b-v0.1",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=8,
        number_nodes=2,
        total_tokens=1_000_000,
    )
    assert r["number_gpus"] == 16
    assert r["train_tokens_per_gpu_per_second"] == pytest.approx(r["train_tokens_per_second"] / 16, rel=1e-9)
    step = simulate_training_step("mistral-7b-v0.1", "NVIDIA-A100-SXM4-80GB", 1024, 4, "full", num_gpus=16, num_nodes=2)
    assert r["train_steps_per_second"] == pytest.approx(
        r["train_tokens_per_second"] / step["tokens_per_step"], rel=1e-9
    )


def test_epochs_times_dataset_equals_total_tokens_path():
    # _resolve_total_tokens: epochs*dataset_tokens must reproduce the same job as an explicit
    # total_tokens of the same product (2 * 500_000 = 1_000_000). Independent path -> same runtime.
    common = dict(
        model_name="mistral-7b-v0.1",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=1,
        number_nodes=1,
    )
    via_epochs = simulate_full_training(**common, epochs=2, dataset_tokens=500_000)
    via_total = simulate_full_training(**common, total_tokens=1_000_000)
    assert via_epochs["total_tokens"] == 1_000_000
    assert via_epochs["train_runtime"] == pytest.approx(via_total["train_runtime"], rel=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 2.0}, "epochs and dataset_tokens together"),  # dataset_tokens missing
        ({"dataset_tokens": 500_000}, "epochs and dataset_tokens together"),  # epochs missing
        ({"epochs": -1.0, "dataset_tokens": 500_000}, "non-negative"),
    ],
)
def test_resolve_total_tokens_error_paths(kwargs, message):
    call = dict(
        model_name="mistral-7b-v0.1",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=1,
        number_nodes=1,
    )
    call.update(kwargs)
    with pytest.raises(ValueError, match=message):
        simulate_full_training(**call)


def test_build_opendc_frames_schema_and_fragments_tile_task_duration():
    # OpenDC interface contract: exact column schemas (a downstream parquet reader needs these
    # names/order) + the tiling invariant -- fragments partition the task duration exactly, each
    # fragment is one simulated step, and all fragments belong to the one task_id.
    tasks, fragments, _summary = build_training_opendc_frames(
        model_name="mistral-7b-v0.1",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=2,
        number_nodes=1,
        total_tokens=1_000_000,
        task_id=1,
        submission_time_ms=1234,
        simulate_full_training_fn=simulate_full_training,
        simulate_training_step_fn=simulate_training_step,
    )

    assert list(tasks.columns) == [
        "id",
        "submission_time",
        "duration",
        "cpu_count",
        "cpu_capacity",
        "mem_capacity",
        "gpu_count",
        "gpu_capacity",
    ]
    assert list(fragments.columns) == ["id", "duration", "cpu_count", "cpu_usage", "gpu_count", "gpu_usage"]
    assert tasks.loc[0, "id"] == 1  # task_id passthrough
    assert tasks.loc[0, "submission_time"] == 1234  # submission_time passthrough
    assert len(fragments) > 1
    assert (fragments["id"] == 1).all()  # every fragment tagged with the task_id
    assert fragments["duration"].nunique() == 1  # uniform steps
    # Tiling: fragments exactly partition the task duration.
    assert fragments["duration"].sum() == tasks.loc[0, "duration"]


def test_export_training_opendc_writes_parquet_with_matching_schema(tmp_path: Path):
    # Exercises the parquet write path; asserts the on-disk schema matches the in-memory contract
    # and that fragment ids round-trip to the given task_id.
    result = export_training_opendc(
        output_dir=str(tmp_path),
        model_name="mistral-7b-v0.1",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=1,
        number_nodes=1,
        total_tokens=100_000,
        task_id=7,
        submission_time_ms=0,
        simulate_full_training_fn=simulate_full_training,
        simulate_training_step_fn=simulate_training_step,
    )
    assert result["train_tokens_per_second"] > 0

    tasks_table = pq.read_table(tmp_path / "tasks.parquet")
    fragments_table = pq.read_table(tmp_path / "fragments.parquet")
    assert tasks_table.column_names == [
        "id",
        "submission_time",
        "duration",
        "cpu_count",
        "cpu_capacity",
        "mem_capacity",
        "gpu_count",
        "gpu_capacity",
    ]
    assert fragments_table.column_names == ["id", "duration", "cpu_count", "cpu_usage", "gpu_count", "gpu_usage"]
    assert tasks_table.num_rows == 1
    assert fragments_table.num_rows > 1
    assert set(fragments_table.column("id").to_pylist()) == {7}  # all fragments carry task_id=7
