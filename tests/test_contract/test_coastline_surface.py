"""Coastline-contract freeze (de-slop Phase 0, Deliverable A).

``../coastline`` (a separate, independently-versioned repo; never imported or modified here) consumes
Kavier through exactly the surfaces exercised below. Call shapes were read directly from that repo, not
guessed:
  - kavier_predictor.py (coastline/src/coastline/sdk/predictors/performance/physics/) calls
    ``kavier.training.performance(row)`` and reads ``train_tokens_per_second``, ``gpu_power_watts``,
    ``step_time_ms`` (expected ABSENT), ``gpu_compute_utilization``, ``gpu_memory_utilization``.
  - workload_queue.py / trace/plot.py (coastline/src/coastline/{ui,sdk/trace}/) call
    ``kavier.sdk.cluster.schedule(...)`` with the two row/kwarg shapes reproduced in item 4 below, and
    read ``result.{dropped,jobs,cluster,timeline}`` and the per-job/per-cluster/per-timeline fields
    asserted there.

This is a characterization/regression suite, not a physics suite (that's owned by test_training/,
test_cluster/, etc.). Every value asserted here is hand-derived in a comment at the point of use; where
a numeric derivation would just duplicate physics already covered elsewhere (e.g. FCFS/backfill timing
math -- see test_cluster/test_schedule_api.py and test_integration/test_cli_cluster.py), this module
sticks to presence/type/shape, per the brief's own "plain asserts for shape/presence" carve-out.

Hard rule for this whole module: if a test here reveals a surprising CURRENT behaviour, it pins that
behaviour and says so in a comment -- it does not "fix" it. src/ is untouched in this phase.
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

# The exact row coastline_predictor.py builds: {"model", "gpu", "method", "seq_len", "batch_size",
# "num_gpus", "num_nodes"} with num_gpus/num_nodes derived so num_gpus is PER NODE (see item 2). Both
# names are calibrated in calibration.json (CLAUDE.md's own example pair), so this must not warn.
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
# 1. kavier.training.performance(row) -- column contract + step_time_ms absence
# ======================================================================================================


def test_performance_exposes_the_columns_coastline_reads() -> None:
    # coastline's KavierPredictor.predict() does result.get("train_tokens_per_second"),
    # .get("gpu_power_watts"), .get("gpu_compute_utilization"), .get("gpu_memory_utilization") and
    # treats a missing key as None rather than KeyError -- so these must be actual columns, not just
    # not-crash. It also asserts throughput > 0 before trusting a prediction.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = kavier_training.performance(COASTLINE_ROW)
    # granite-3-8b / NVIDIA-A100-SXM4-80GB is a calibrated pair (CLAUDE.md's own example) -- a
    # fallback-to-neutral-1.0 warning here would mean the shipped calibration table lost this pair.
    # Scoped to calibration warnings only, so an unrelated pandas/numpy deprecation can't flip this.
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
        assert math.isfinite(row[col])  # rejects NaN and ±inf outright
        assert row[col] > 0

    # memory utilization is a physical percentage; unlike the above it can legitimately be 0, never <0.
    assert "gpu_memory_utilization" in df.columns
    assert row["gpu_memory_utilization"] >= 0

    # total_tokens: COASTLINE_ROW carries no total_tokens/epochs/dataset_tokens key, so
    # core/engine.py::_resolve_total_tokens has nothing to resolve from and returns None by contract
    # ("total_tokens (wins) or round(epochs * dataset_tokens); None if neither").
    assert "total_tokens" in df.columns
    assert row["total_tokens"] is None

    # train_runtime = total_tokens / tps if total_tokens is not None else 0.0 (core/engine.py). Since
    # total_tokens is None here, this is exactly 0.0 -- pinned exactly, not just "finite/small", because
    # it is a deliberate branch, not an approximation. Coastline itself documents this: its predictor
    # hardcodes predicted_runtime_seconds=None with the comment "Kavier yields per-step time, not job
    # runtime" -- i.e. it never reads train_runtime at all, consistent with it being a placeholder here.
    assert "train_runtime" in df.columns
    assert row["train_runtime"] == 0.0

    # Coastline's own comment: "step_time_ms ... not exported by the verb -> None". facade.py::performance
    # selects a fixed `cols` allowlist that omits it on purpose (it's only in the internal step dict).
    assert "step_time_ms" not in df.columns


# ======================================================================================================
# 2. num_gpus is PER-NODE: total billed GPUs = num_gpus * num_nodes
# ======================================================================================================


def test_num_gpus_is_per_node_aggregate_power_doubles_with_num_nodes() -> None:
    # facade.py::run_training: total_gpus = int(p["num_gpus"]) * int(p["num_nodes"]); performance()'s
    # gpu_power_watts column is simulate_training_step's per-GPU watts (run_training only multiplies by
    # total_gpus into the separate, non-exported "aggregate_power_w" key) -- confirmed by reading the
    # facade before writing this oracle, per the brief. So "aggregate power" must be reconstructed here
    # from the public columns, using the num_gpus/num_nodes WE passed in.
    row_1_node = COASTLINE_ROW
    row_2_nodes = {**COASTLINE_ROW, "num_nodes": 2}

    r1 = kavier_training.performance(row_1_node).iloc[0]
    r2 = kavier_training.performance(row_2_nodes).iloc[0]

    # core/engine.py::_compute_mfu(batch_size, gpu, calibrated) takes no num_gpus/num_nodes argument, so
    # gpu_compute_utilization (= mfu * 100) MUST be bit-identical regardless of node count -- an exact
    # hand-derived invariant from the function's own parameter list, not an empirical observation.
    assert r1["gpu_compute_utilization"] == r2["gpu_compute_utilization"]

    # core/engine.py::_comm_time: num_nodes<=1 uses one ring-allreduce term; num_nodes>1 uses that SAME
    # intra-node term (gpus_per_node = num_gpus/num_nodes = 8 in both configs here) PLUS an extra
    # inter-node term -- so step_time_s can only be >= the 1-node case. Memory utilization is
    # traffic / step_time_s (_estimate_memory_bandwidth_usage) with fixed traffic, so a larger step_time_s
    # can only push memory utilization DOWN, never up.
    assert r2["gpu_memory_utilization"] <= r1["gpu_memory_utilization"]
    # mse_power's u = max(compute_util, memory_util) (energy/engine.py::mse_power). memory_util is far
    # below compute_util at 1 node already (single-digit % vs ~40%, read off this exact row while writing
    # this test) and can only fall further at 2 nodes (previous assert), so u == mfu at BOTH node counts
    # and stays there -- meaning per-GPU watts must be exactly equal, not merely close.
    assert r1["gpu_compute_utilization"] > r1["gpu_memory_utilization"]
    assert r1["gpu_power_watts"] == r2["gpu_power_watts"]

    total_gpus_1 = row_1_node["num_gpus"] * row_1_node["num_nodes"]  # 8 * 1 = 8
    total_gpus_2 = row_2_nodes["num_gpus"] * row_2_nodes["num_nodes"]  # 8 * 2 = 16
    assert total_gpus_2 == 2 * total_gpus_1

    aggregate_1 = r1["gpu_power_watts"] * total_gpus_1
    aggregate_2 = r2["gpu_power_watts"] * total_gpus_2
    # Per-GPU watts identical (just shown) and billed GPUs exactly double -> aggregate exactly doubles.
    assert aggregate_2 == pytest.approx(2.0 * aggregate_1)


# ======================================================================================================
# 3. Exception contract: unknown model / unknown gpu -> KeyError
# ======================================================================================================


def test_unknown_model_raises_key_error() -> None:
    # UnknownSpecError(KeyError) from library/lookup.py::get_llm, reached via simulate_training_step.
    with pytest.raises(KeyError):
        kavier_training.performance({**COASTLINE_ROW, "model": "not-a-real-model"})


def test_unknown_gpu_raises_key_error() -> None:
    with pytest.raises(KeyError):
        kavier_training.performance({**COASTLINE_ROW, "gpu": "not-a-real-gpu"})


# ======================================================================================================
# 4. kavier.sdk.cluster.schedule -- the two call shapes Coastline uses
# ======================================================================================================


def test_schedule_fcfs_shape_matches_coastline_workload_queue() -> None:
    # Mirrors coastline/src/coastline/ui/workload_queue.py::simulate_fifo exactly: rows keyed
    # job_id/submit_s/gpus/duration_s/power_w_per_gpu, policy="distributed-fcfs", oversized="drop",
    # default_watts_per_gpu set (that file's _AVG_WATTS_PER_GPU). "big" wants more GPUs (999) than the
    # 32-GPU pool -> dropped, matching the oversized="drop" contract (already physics-tested by
    # test_cluster/test_schedule_api.py::test_oversized_drop_excludes_the_job_and_reports_it; here we
    # only confirm this exact call shape still drops + that every field Coastline reads is present).
    rows = [
        {"job_id": "big", "submit_s": 0.0, "gpus": 999, "duration_s": 10.0, "power_w_per_gpu": 300.0},
        {"job_id": "j1", "submit_s": 0.0, "gpus": 4, "duration_s": 10.0, "power_w_per_gpu": 300.0},
        {"job_id": "j2", "submit_s": 0.0, "gpus": 4, "duration_s": 10.0},  # no power -> default applies
    ]
    result = schedule(
        rows, policy="distributed-fcfs", num_nodes=1, node_gpus=32, oversized="drop", default_watts_per_gpu=350.0
    )

    assert result.dropped == ["big"]
    assert len(result.jobs) == 2

    # j1 and j2 both fit (4+4=8 GPUs on a 32-GPU pool) and submit at the same instant -> both start
    # immediately, no queueing: start=submit=0, end=0+duration=10, wait=0, turnaround=end-submit=10.
    by_id = {j.job_id: j for j in result.jobs}
    j1, j2 = by_id["j1"], by_id["j2"]
    assert (j1.gpus, j1.submit_s, j1.start_s, j1.end_s, j1.wait_s, j1.runtime_s, j1.turnaround_s) == (
        4, 0.0, 0.0, 10.0, 0.0, 10.0, 10.0,
    )  # fmt: skip
    # energy_kwh = power_w_per_gpu * gpus * runtime_s / 3.6e6 (facade.py::schedule docstring).
    assert j1.energy_kwh == pytest.approx(300.0 * 4 * 10.0 / 3.6e6)  # = 0.003333...
    # j2 has no power_w_per_gpu -> falls back to default_watts_per_gpu=350.0.
    assert j2.energy_kwh == pytest.approx(350.0 * 4 * 10.0 / 3.6e6)  # = 0.003888...

    c = result.cluster
    assert c.n_jobs == 2
    assert c.makespan_s == pytest.approx(10.0)  # both jobs occupy exactly [0, 10]
    assert c.avg_wait_s == pytest.approx(0.0)
    assert c.avg_turnaround_s == pytest.approx(10.0)
    # utilization = gpu-seconds used / (capacity * makespan) = (4*10 + 4*10) / (32*10) = 80/320 = 0.25.
    assert c.utilization == pytest.approx(0.25)
    assert c.goodput_jobs_per_s == pytest.approx(2 / 10.0)
    assert c.total_energy_kwh == pytest.approx(j1.energy_kwh + j2.energy_kwh)
    assert c.n_jobs == 2
    assert c.peak_gpus == 8  # j1+j2 run concurrently: 4+4
    assert c.peak_queue == 0  # neither job ever waits

    # Sane-type presence for every attribute coastline's workload_queue.py reads off a JobRecord.
    for job in result.jobs:
        assert isinstance(job.job_id, str)
        assert isinstance(job.gpus, int)
        for attr in ("runtime_s", "submit_s", "start_s", "end_s", "wait_s", "turnaround_s"):
            assert isinstance(getattr(job, attr), float)
        assert job.energy_kwh is None or isinstance(job.energy_kwh, float)


def test_schedule_backfill_shape_matches_coastline_trace_plot() -> None:
    # Mirrors coastline/src/coastline/sdk/trace/plot.py exactly: rows keyed
    # submit_s/gpus/duration_s/nodes (nodes is accepted but IGNORED by the scheduler -- placement is
    # automatic tight-pack, per facade.py::_normalise's own docstring), policy="distributed-backfill", no oversized/
    # default_watts_per_gpu override (so the "cap"/None defaults apply).
    rows = [
        {"submit_s": 0.0, "gpus": 8, "duration_s": 20.0, "nodes": 1},
        {"submit_s": 5.0, "gpus": 4, "duration_s": 5.0, "nodes": 1},
    ]
    result = schedule(rows, policy="distributed-backfill", num_nodes=2, node_gpus=8)

    assert result.dropped == []
    assert len(result.jobs) == 2
    j0, j1 = result.jobs  # tight-pack: job0 (8 GPUs) claims node 0 entirely at t=0.
    assert (j0.gpus, j0.submit_s, j0.start_s, j0.end_s, j0.wait_s, j0.turnaround_s) == (
        8, 0.0, 0.0, 20.0, 0.0, 20.0,
    )  # fmt: skip
    # job1 (4 GPUs) submits at t=5 while node 0 is full; node 1 is still free -> backfills immediately
    # (this is the behaviour the distributed-backfill policy is being relied on for): start=submit=5, wait=0.
    assert (j1.gpus, j1.submit_s, j1.start_s, j1.end_s, j1.wait_s, j1.turnaround_s) == (
        4, 5.0, 5.0, 10.0, 0.0, 5.0,
    )  # fmt: skip
    # No power_w_per_gpu column and no default_watts_per_gpu kwarg -> energy is genuinely unknown (None),
    # not 0 -- this is the "never a poisoned NaN/0" contract from facade.py's _node_records docstring.
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
    assert c.total_energy_kwh is None  # no known per-job energy anywhere -> cluster total is unknown too
    assert c.peak_gpus == 12  # job0 (8) and job1 (4) overlap on [5, 10] -> 8+4
    assert c.peak_queue == 0

    # Sane-type presence for the Timeline attributes coastline's trace/plot.py reads directly.
    t = result.timeline
    assert isinstance(t.times_h, list) and isinstance(t.gpus_in_use, list) and isinstance(t.queue_depth, list)
    assert len(t.times_h) == len(t.gpus_in_use) == len(t.queue_depth)
    assert all(isinstance(x, float) for x in t.times_h)
    assert c.makespan_h == pytest.approx(c.makespan_s / 3600.0)
    assert c.peak_queue >= 0 and c.peak_gpus >= 0


# ======================================================================================================
# 5. kavier.sdk.library -- get_llm/get_gpu/GPU_SPEC_LIBRARY/LLM_SPEC_LIBRARY importable + spec attrs
# ======================================================================================================


@pytest.mark.parametrize("attr", ["m_params", "active_params", "n_layers", "d_model", "n_heads", "d_head"])
def test_llm_spec_has_the_attrs_coastline_reads(attr: str) -> None:
    llm = get_llm("granite-3-8b")
    value = getattr(llm, attr)  # AttributeError here IS the failure this test exists to catch
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
    # kavier.sdk.library.__init__ exports exactly these four names (read before writing this test).
    assert get_llm("granite-3-8b") is LLM_SPEC_LIBRARY["granite-3-8b"]
    assert get_gpu("NVIDIA-A100-SXM4-80GB") is GPU_SPEC_LIBRARY["NVIDIA-A100-SXM4-80GB"]


# ======================================================================================================
# 6. Signature pins -- these kwarg names are public and must never silently drift
# ======================================================================================================


def test_simulate_training_step_signature_is_pinned() -> None:
    # Read verbatim from core/engine.py::simulate_training_step's def line; confirmed via
    # inspect.signature while writing this test (not guessed).
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
    # NOTE (surprising but real): simulate_full_training's parameter ORDER and NAMES diverge from
    # simulate_training_step's -- (model_name, method, gpu_model, ...) here vs (model_name, gpu_model,
    # ..., method) there, and number_gpus/number_nodes here vs num_gpus/num_nodes there. Pinned as-is;
    # not this phase's job to reconcile it (flagged in the Phase 0 report).
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
# 7. Packaged data -- calibration.json ships inside the wheel and is valid JSON
# ======================================================================================================


def test_calibration_json_is_packaged_and_parses() -> None:
    resource = files("kavier.sdk.training").joinpath("calibration", "calibration.json")
    assert resource.is_file()
    data = json.loads(resource.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data  # non-empty


# ======================================================================================================
# 8. Calibration selector surface -- names exist; selector LOGIC is owned by test_calibration_versions.py
# ======================================================================================================


def test_calibration_selector_names_exist() -> None:
    from kavier.sdk.training import calibration

    names = calibration.available_calibrations()
    assert isinstance(names, list)
    assert len(names) > 0
    assert callable(calibration.use_calibration)
    # $KAVIER_CALIBRATION is read via os.environ.get(_ENV_VAR) inside the module; confirm the literal
    # env-var name is really the one the module reads, without re-testing the selection logic itself.
    assert "KAVIER_CALIBRATION" in inspect.getsource(calibration)
