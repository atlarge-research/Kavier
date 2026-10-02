"""Pin the Kavier API used by Coastline, a separately versioned repository.

Call sites in coastline/src/coastline/:
  - sdk/predictors/performance/physics/kavier_predictor.py calls ``kavier.training.performance(row)``
    and reads train_tokens_per_second, gpu_power_watts, gpu_compute_utilization,
    gpu_memory_utilization, and step_time_ms (expected absent).
  - ui/workload_queue.py and sdk/trace/plot.py call ``kavier.sdk.cluster.schedule(...)`` with the two
    shapes in section 4 and read ``result.{dropped,jobs,cluster,timeline}``.

These are regression tests of current behaviour; the physics is tested in test_training/ and
test_cluster/. Where a value would repeat scheduler timing tests (test_cluster/test_schedule_api.py,
test_integration/test_cli_cluster.py), only presence, type and shape are checked.
"""

from __future__ import annotations

import inspect
import json
import math
import warnings
from importlib.resources import files

import pytest

from kavier import training as kavier_training
from kavier.sdk.cluster import schedule
from kavier.sdk.library import GPU_SPEC_LIBRARY, LLM_SPEC_LIBRARY, get_gpu, get_llm
from kavier.sdk.training.core.engine import simulate_full_training, simulate_training_step

# The row kavier_predictor.py builds; num_gpus is per node (section 2).
# granite-3-8b and NVIDIA-A100-SXM4-80GB are both calibrated in calibration.json, so no warning.
COASTLINE_ROW = {
    "model": "granite-3-8b",
    "gpu": "NVIDIA-A100-SXM4-80GB",
    "method": "full",
    "seq_len": 1024,
    "batch_size": 4,
    "num_gpus": 8,
    "num_nodes": 1,
}


# ======================================================================================================
# 1. kavier.training.performance(row): exported columns, no step_time_ms
# ======================================================================================================


def test_performance_exposes_the_columns_coastline_reads() -> None:
    # KavierPredictor.predict() reads these with result.get(), so a missing column becomes None
    # silently. It also requires throughput > 0.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = kavier_training.performance(COASTLINE_ROW)
    # A fallback-to-1.0 warning would mean calibration.json lost this pair. Only calibration
    # warnings count; unrelated deprecation warnings are ignored.
    assert [w for w in caught if "calibration" in str(w.message).lower()] == []

    assert len(df) == 1
    row = df.iloc[0]

    for col in (
        "train_tokens_per_second",
        "gpu_power_watts",
        "gpu_compute_utilization",
        "train_samples_per_second",
    ):
        assert col in df.columns
        assert math.isfinite(row[col])
        assert row[col] > 0

    # Memory utilization is a percentage and can be 0.
    assert "gpu_memory_utilization" in df.columns
    assert row["gpu_memory_utilization"] >= 0

    # COASTLINE_ROW has no total_tokens, epochs or dataset_tokens, so
    # core/engine.py::_resolve_total_tokens returns None.
    assert "total_tokens" in df.columns
    assert row["total_tokens"] is None

    # core/engine.py sets train_runtime = 0.0 when total_tokens is None. Coastline does not read
    # train_runtime; its predictor sets predicted_runtime_seconds=None.
    assert "train_runtime" in df.columns
    assert row["train_runtime"] == 0.0

    # facade.py::performance exports a fixed `cols` list; step_time_ms is only in the internal step dict.
    assert "step_time_ms" not in df.columns


# ======================================================================================================
# 2. num_gpus is per node: total billed GPUs = num_gpus * num_nodes
# ======================================================================================================


def test_num_gpus_is_per_node_aggregate_power_doubles_with_num_nodes() -> None:
    # facade.py::run_training: total_gpus = num_gpus * num_nodes. The exported gpu_power_watts is
    # per GPU; "aggregate_power_w" is not exported, so aggregate power is rebuilt from the inputs here.
    row_1_node = COASTLINE_ROW
    row_2_nodes = {**COASTLINE_ROW, "num_nodes": 2}

    r1 = kavier_training.performance(row_1_node).iloc[0]
    r2 = kavier_training.performance(row_2_nodes).iloc[0]

    # core/engine.py::_compute_mfu(batch_size, gpu, calibrated) takes no GPU or node count, so
    # gpu_compute_utilization (mfu * 100) is identical for both node counts.
    assert r1["gpu_compute_utilization"] == r2["gpu_compute_utilization"]

    # core/engine.py::_comm_time: num_nodes > 1 adds an inter-node term to the same intra-node term
    # (8 GPUs per node in both configs), so step_time_s does not decrease. Memory utilization is
    # traffic / step_time_s with fixed traffic (_estimate_memory_bandwidth_usage), so it does not increase.
    assert r2["gpu_memory_utilization"] <= r1["gpu_memory_utilization"]
    # energy/engine.py::mse_power uses u = max(compute_util, memory_util). memory_util is single-digit %
    # against ~40% compute at 1 node and falls at 2 nodes, so u = mfu at both and per-GPU watts are equal.
    assert r1["gpu_compute_utilization"] > r1["gpu_memory_utilization"]
    assert r1["gpu_power_watts"] == r2["gpu_power_watts"]

    total_gpus_1 = row_1_node["num_gpus"] * row_1_node["num_nodes"]  # 8 * 1 = 8
    total_gpus_2 = row_2_nodes["num_gpus"] * row_2_nodes["num_nodes"]  # 8 * 2 = 16
    assert total_gpus_2 == 2 * total_gpus_1

    aggregate_1 = r1["gpu_power_watts"] * total_gpus_1
    aggregate_2 = r2["gpu_power_watts"] * total_gpus_2
    # Equal per-GPU watts and twice the GPUs give twice the aggregate.
    assert aggregate_2 == pytest.approx(2.0 * aggregate_1)


# ======================================================================================================
# 3. Unknown model or GPU raises KeyError
# ======================================================================================================


def test_unknown_model_raises_key_error() -> None:
    # UnknownSpecError(KeyError) from library/lookup.py::get_llm via simulate_training_step.
    with pytest.raises(KeyError):
        kavier_training.performance({**COASTLINE_ROW, "model": "not-a-real-model"})


def test_unknown_gpu_raises_key_error() -> None:
    with pytest.raises(KeyError):
        kavier_training.performance({**COASTLINE_ROW, "gpu": "not-a-real-gpu"})


# ======================================================================================================
# 4. kavier.sdk.cluster.schedule: the two call shapes Coastline uses
# ======================================================================================================


def test_schedule_fcfs_shape_matches_coastline_workload_queue() -> None:
    # Call shape of coastline/src/coastline/ui/workload_queue.py::simulate_fifo, with
    # default_watts_per_gpu = _AVG_WATTS_PER_GPU. "big" asks for 999 GPUs on a 32-GPU pool and is
    # dropped; drop behaviour is tested in test_cluster/test_schedule_api.py.
    rows = [
        {"job_id": "big", "submit_s": 0.0, "gpus": 999, "duration_s": 10.0, "power_w_per_gpu": 300.0},
        {"job_id": "j1", "submit_s": 0.0, "gpus": 4, "duration_s": 10.0, "power_w_per_gpu": 300.0},
        {"job_id": "j2", "submit_s": 0.0, "gpus": 4, "duration_s": 10.0},  # default power applies
    ]
    result = schedule(
        rows, policy="distributed-fcfs", num_nodes=1, node_gpus=32, oversized="drop", default_watts_per_gpu=350.0
    )

    assert result.dropped == ["big"]
    assert len(result.jobs) == 2

    # j1 and j2 fit together (8 of 32 GPUs) and start at submit: start=0, end=10, wait=0, turnaround=10.
    by_id = {j.job_id: j for j in result.jobs}
    j1, j2 = by_id["j1"], by_id["j2"]
    assert (j1.gpus, j1.submit_s, j1.start_s, j1.end_s, j1.wait_s, j1.runtime_s, j1.turnaround_s) == (
        4, 0.0, 0.0, 10.0, 0.0, 10.0, 10.0,
    )  # fmt: skip
    # energy_kwh = power_w_per_gpu * gpus * runtime_s / 3.6e6 (facade.py::schedule)
    assert j1.energy_kwh == pytest.approx(300.0 * 4 * 10.0 / 3.6e6)  # = 0.003333...
    # j2 uses default_watts_per_gpu=350.0.
    assert j2.energy_kwh == pytest.approx(350.0 * 4 * 10.0 / 3.6e6)  # = 0.003888...

    c = result.cluster
    assert c.n_jobs == 2
    assert c.makespan_s == pytest.approx(10.0)  # both jobs occupy exactly [0, 10]
    assert c.avg_wait_s == pytest.approx(0.0)
    assert c.avg_turnaround_s == pytest.approx(10.0)
    # utilization = gpu-seconds / (capacity * makespan) = (4*10 + 4*10) / (32*10) = 0.25
    assert c.utilization == pytest.approx(0.25)
    assert c.goodput_jobs_per_s == pytest.approx(2 / 10.0)
    assert c.total_energy_kwh == pytest.approx(j1.energy_kwh + j2.energy_kwh)
    assert c.n_jobs == 2
    assert c.peak_gpus == 8  # j1+j2 run concurrently: 4+4
    assert c.peak_queue == 0

    # Types of the JobRecord attributes workload_queue.py reads.
    for job in result.jobs:
        assert isinstance(job.job_id, str)
        assert isinstance(job.gpus, int)
        for attr in ("runtime_s", "submit_s", "start_s", "end_s", "wait_s", "turnaround_s"):
            assert isinstance(getattr(job, attr), float)
        assert job.energy_kwh is None or isinstance(job.energy_kwh, float)


def test_schedule_backfill_shape_matches_coastline_trace_plot() -> None:
    # Call shape of coastline/src/coastline/sdk/trace/plot.py. The scheduler ignores "nodes" and
    # tight-packs (facade.py::_normalise). oversized and default_watts_per_gpu keep their defaults.
    rows = [
        {"submit_s": 0.0, "gpus": 8, "duration_s": 20.0, "nodes": 1},
        {"submit_s": 5.0, "gpus": 4, "duration_s": 5.0, "nodes": 1},
    ]
    result = schedule(rows, policy="distributed-backfill", num_nodes=2, node_gpus=8)

    assert result.dropped == []
    assert len(result.jobs) == 2
    j0, j1 = result.jobs  # job0 (8 GPUs) fills node 0 at t=0
    assert (j0.gpus, j0.submit_s, j0.start_s, j0.end_s, j0.wait_s, j0.turnaround_s) == (
        8, 0.0, 0.0, 20.0, 0.0, 20.0,
    )  # fmt: skip
    # job1 (4 GPUs) submits at t=5 with node 0 full and starts on node 1: start=5, wait=0.
    assert (j1.gpus, j1.submit_s, j1.start_s, j1.end_s, j1.wait_s, j1.turnaround_s) == (
        4, 5.0, 5.0, 10.0, 0.0, 5.0,
    )  # fmt: skip
    # Without power_w_per_gpu or default_watts_per_gpu, energy is None (facade.py::_node_records).
    assert j0.energy_kwh is None and j1.energy_kwh is None

    c = result.cluster
    assert c.n_jobs == 2
    assert c.capacity_gpus == 16  # num_nodes(2) * node_gpus(8)
    assert c.makespan_s == pytest.approx(20.0)  # max(end) - min(start) = 20 - 0
    assert c.avg_wait_s == pytest.approx(0.0)
    assert c.avg_turnaround_s == pytest.approx((20.0 + 5.0) / 2)  # = 12.5
    # utilization = (8*20 + 4*5) / (16*20) = 180/320 = 0.5625
    assert c.utilization == pytest.approx(0.5625)
    assert c.goodput_jobs_per_s == pytest.approx(2 / 20.0)
    assert c.total_energy_kwh is None
    assert c.peak_gpus == 12  # job0 (8) and job1 (4) overlap on [5, 10]
    assert c.peak_queue == 0

    # Types of the Timeline attributes trace/plot.py reads.
    t = result.timeline
    assert isinstance(t.times_h, list) and isinstance(t.gpus_in_use, list) and isinstance(t.queue_depth, list)
    assert len(t.times_h) == len(t.gpus_in_use) == len(t.queue_depth)
    assert all(isinstance(x, float) for x in t.times_h)
    assert c.makespan_h == pytest.approx(c.makespan_s / 3600.0)
    assert c.peak_queue >= 0 and c.peak_gpus >= 0


# ======================================================================================================
# 5. kavier.sdk.library: get_llm, get_gpu, GPU_SPEC_LIBRARY, LLM_SPEC_LIBRARY and spec attributes
# ======================================================================================================


@pytest.mark.parametrize("attr", ["m_params", "active_params", "n_layers", "d_model", "n_heads", "d_head"])
def test_llm_spec_has_the_attrs_coastline_reads(attr: str) -> None:
    llm = get_llm("granite-3-8b")
    value = getattr(llm, attr)
    assert isinstance(value, (int, float))
    assert value > 0


@pytest.mark.parametrize(
    "attr",
    [
        "memory_gb",
        "fp_16_tensor_core_tflops",
        "bandwidth_bps",
        "max_power_w",
        "cores",
        "core_max_mhz",
        "base_power_w",
        "network_bandwidth_gbps",
    ],
)
def test_gpu_spec_has_the_attrs_coastline_reads(attr: str) -> None:
    gpu = get_gpu("NVIDIA-A100-SXM4-80GB")
    value = getattr(gpu, attr)
    assert isinstance(value, (int, float))
    assert value > 0


def test_library_top_level_names_importable() -> None:
    # kavier.sdk.library exports these four names.
    assert get_llm("granite-3-8b") is LLM_SPEC_LIBRARY["granite-3-8b"]
    assert get_gpu("NVIDIA-A100-SXM4-80GB") is GPU_SPEC_LIBRARY["NVIDIA-A100-SXM4-80GB"]


# ======================================================================================================
# 6. Signature pins: these keyword names are public
# ======================================================================================================


def test_simulate_training_step_signature_is_pinned() -> None:
    assert tuple(inspect.signature(simulate_training_step).parameters) == (
        "model_name",
        "gpu_model",
        "tokens_per_sample",
        "batch_size",
        "method",
        "num_gpus",
        "num_nodes",
        "grad_accum_steps",
        "backward_factor",
        "calibrated",
    )


def test_simulate_full_training_signature_is_pinned() -> None:
    # Parameter order and names differ from simulate_training_step: method comes second here, and
    # number_gpus/number_nodes replace num_gpus/num_nodes. Pinned as is.
    assert tuple(inspect.signature(simulate_full_training).parameters) == (
        "model_name",
        "method",
        "gpu_model",
        "tokens_per_sample",
        "batch_size",
        "number_gpus",
        "number_nodes",
        "total_tokens",
        "epochs",
        "dataset_tokens",
        "grad_accum_steps",
        "backward_factor",
    )


# ======================================================================================================
# 7. calibration.json ships in the wheel and parses
# ======================================================================================================


def test_calibration_json_is_packaged_and_parses() -> None:
    resource = files("kavier.sdk.training").joinpath("calibration", "calibration.json")
    assert resource.is_file()
    data = json.loads(resource.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data  # non-empty


# ======================================================================================================
# 8. Calibration selector names exist; selection logic is tested in test_calibration_versions.py
# ======================================================================================================


def test_calibration_selector_names_exist() -> None:
    from kavier.sdk.training import calibration

    names = calibration.available_calibrations()
    assert isinstance(names, list)
    assert len(names) > 0
    assert callable(calibration.use_calibration)
    # The module reads $KAVIER_CALIBRATION through os.environ.get(_ENV_VAR).
    assert "KAVIER_CALIBRATION" in inspect.getsource(calibration)
