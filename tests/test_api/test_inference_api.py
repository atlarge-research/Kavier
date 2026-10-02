"""Tests for the ``kavier.inference`` batch API: output shape, input forms, and analytic checks.

The facade feeds a single-stream roofline simulation (``run_inference``) into carbon, energy and cost
on a flat trace. The numeric tests take ``total_s`` from ``performance`` and recompute energy, carbon
and cost in closed form for constant power and intensity, apart from ``Fragment``/``compute_emissions``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kavier
from kavier.sdk.inference import (
    DEFAULT_GPU_HOUR_PRICE,
    DEFAULT_INTENSITY_G_KWH,
    run_inference,
)
from kavier.sdk.library import get_gpu

ROW_A = {"model": "Llama-3-8B", "gpu": "A10", "num_requests": 16, "input_tokens": 128, "output_tokens": 32}
ROW_B = {
    "model": "mistral-7b-v0.1",
    "gpu": "NVIDIA-A100-SXM4-80GB",
    "num_requests": 8,
    "input_tokens": 256,
    "output_tokens": 64,
}

# n identical requests, each with (in + out) tokens.
TOKENS_A = ROW_A["num_requests"] * (ROW_A["input_tokens"] + ROW_A["output_tokens"])  # 16*160 = 2560


def _batch() -> pd.DataFrame:
    return pd.DataFrame([ROW_A, ROW_B])


@pytest.fixture(scope="module")
def perf_a() -> "pd.Series":
    """Return the ``performance`` row for ROW_A; its ``total_s`` feeds the energy, carbon and cost checks."""
    return kavier.inference.performance(ROW_A).iloc[0]


# ---------------------------------------------------------------------------------------------------
# Output shape and input forms
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verb", "expected_cols"),
    [
        (kavier.inference.performance, ("p50_ms", "p95_ms", "mean_ttft_ms", "throughput_tok_s", "total_s")),
        (kavier.inference.energy, ("energy_wh", "energy_per_mtoken_wh")),
        (kavier.inference.efficiency, ("financial_per_mtoken",)),
        (kavier.inference.carbon, ("carbon_per_mtoken_g", "total_co2_g")),
    ],
)
def test_verb_returns_row_per_workload_with_predicted_cols_and_preserved_input(verb, expected_cols) -> None:
    # One row per input row in order, input columns kept, documented columns present.
    batch = _batch()
    out = verb(batch)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(batch)  # one output row per input row, order preserved
    assert list(out["model"]) == [ROW_A["model"], ROW_B["model"]]
    assert list(out["gpu"]) == [ROW_A["gpu"], ROW_B["gpu"]]
    for col in expected_cols:
        assert col in out.columns


@pytest.mark.parametrize(
    "verb",
    [kavier.inference.performance, kavier.inference.energy, kavier.inference.efficiency, kavier.inference.carbon],
)
def test_single_dict_list_and_dataframe_are_equivalent_inputs(verb) -> None:
    # _normalise treats a DataFrame, a list and a dict the same way.
    from_df = verb(pd.DataFrame([ROW_A])).iloc[0]
    from_list = verb([ROW_A]).iloc[0]
    from_dict = verb(ROW_A).iloc[0]
    pd.testing.assert_series_equal(from_df, from_list, check_names=False)
    pd.testing.assert_series_equal(from_df, from_dict, check_names=False)


@pytest.mark.parametrize(
    "bad",
    [
        {**ROW_A, "num_requests": 0},
        {**ROW_A, "input_tokens": -1},
        {**ROW_A, "output_tokens": -5},
    ],
)
def test_unsimulatable_workload_is_rejected(bad) -> None:
    # _infer_params raises on zero requests or negative token counts.
    with pytest.raises(ValueError):
        kavier.inference.performance(bad)


# ---------------------------------------------------------------------------------------------------
# performance: single-stream properties (timing physics is tested in test_inference)
# ---------------------------------------------------------------------------------------------------


def test_throughput_is_total_tokens_over_total_time(perf_a) -> None:
    # throughput = total_tokens / total_s.
    assert perf_a["throughput_tok_s"] == pytest.approx(perf_a["total_tokens"] / perf_a["total_s"])
    assert perf_a["total_tokens"] == TOKENS_A


def test_identical_requests_give_equal_percentiles_and_evenly_split_latency(perf_a) -> None:
    # Single stream, no contention: 16 identical requests have equal latency, so all percentiles
    # equal total_s / num_requests * 1000.
    assert perf_a["p50_ms"] == pytest.approx(perf_a["p95_ms"])
    assert perf_a["p50_ms"] == pytest.approx(perf_a["total_s"] / ROW_A["num_requests"] * 1000.0)


def test_mean_ttft_is_the_prefill_only_share_below_full_latency(perf_a) -> None:
    # TTFT is prefill time; full latency adds the 32 decode tokens, so 0 < ttft < p50.
    assert 0.0 < perf_a["mean_ttft_ms"] < perf_a["p50_ms"]


# ---------------------------------------------------------------------------------------------------
# energy / carbon / cost: closed form at constant power and intensity
# ---------------------------------------------------------------------------------------------------


def test_energy_bills_gpu_max_power_over_busy_time(perf_a) -> None:
    total_s = perf_a["total_s"]
    max_power_w = get_gpu(ROW_A["gpu"]).max_power_w  # 150 W for the A10
    # Constant power P over time t: Wh = P*t/3600.
    expected_wh = max_power_w * total_s / 3600.0
    out = kavier.inference.energy(ROW_A).iloc[0]
    assert out["energy_wh"] == pytest.approx(expected_wh)
    assert out["energy_per_mtoken_wh"] == pytest.approx(expected_wh * 1_000_000.0 / TOKENS_A)


def test_carbon_is_energy_kwh_times_grid_intensity(perf_a) -> None:
    total_s = perf_a["total_s"]
    max_power_w = get_gpu(ROW_A["gpu"]).max_power_w
    # A flat trace has no down-estimation: gCO2 = (P*t / 3.6e6 kWh) * intensity. Default 400 g/kWh.
    expected_co2_g = max_power_w * total_s / 3.6e6 * DEFAULT_INTENSITY_G_KWH
    out = kavier.inference.carbon(ROW_A).iloc[0]
    assert out["total_co2_g"] == pytest.approx(expected_co2_g)
    assert out["carbon_per_mtoken_g"] == pytest.approx(expected_co2_g * 1_000_000.0 / TOKENS_A)


def test_carbon_scales_linearly_with_the_intensity_column() -> None:
    # Doubling grid intensity doubles emissions; the `intensity` column overrides the 400 default.
    base = kavier.inference.carbon({**ROW_A, "intensity": 400.0}).iloc[0]["total_co2_g"]
    doubled = kavier.inference.carbon({**ROW_A, "intensity": 800.0}).iloc[0]["total_co2_g"]
    assert doubled == pytest.approx(2.0 * base)


def test_cost_is_gpu_hours_times_price_with_column_override(perf_a) -> None:
    gpu_hours = perf_a["total_s"] / 3600.0
    # $/Mtoken = gpu_hours * $/hr * 1e6/tokens. Default price 2.5; a gpu_hour_price column overrides it.
    expected_default = gpu_hours * DEFAULT_GPU_HOUR_PRICE * 1_000_000.0 / TOKENS_A
    out_default = kavier.inference.efficiency(ROW_A).iloc[0]
    assert out_default["financial_per_mtoken"] == pytest.approx(expected_default)

    out_priced = kavier.inference.efficiency({**ROW_A, "gpu_hour_price": 5.0}).iloc[0]
    assert out_priced["financial_per_mtoken"] == pytest.approx(gpu_hours * 5.0 * 1_000_000.0 / TOKENS_A)


# ---------------------------------------------------------------------------------------------------
# Facade default parameters
# ---------------------------------------------------------------------------------------------------


def test_facade_defaults_to_kv_cache_on(perf_a) -> None:
    # Default kv_cache=True: matches an explicit kv_cache=True run and differs from kv_cache=False.
    base = {**ROW_A, "prefix_policy": "none", "prefix_min_tokens": 1024}
    on = run_inference({**base, "kv_cache": True})
    off = run_inference({**base, "kv_cache": False})
    assert perf_a["total_s"] == pytest.approx(on["total_s"])
    assert off["total_s"] > on["total_s"]  # quadratic decode without KV cache


def test_unknown_prefix_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="bogus"):
        kavier.inference.performance({**ROW_A, "prefix_policy": "bogus"})


@pytest.mark.parametrize(("given", "same_as"), [("on", True), ("off", False), ("True", True), ("false", False)])
def test_kv_cache_strings_map_to_on_and_off(given, same_as) -> None:
    from_string = kavier.inference.performance({**ROW_A, "kv_cache": given}).iloc[0]["total_s"]
    from_bool = kavier.inference.performance({**ROW_A, "kv_cache": same_as}).iloc[0]["total_s"]
    assert from_string == from_bool


@pytest.mark.parametrize("bad", ["maybe", "", 2])
def test_unknown_kv_cache_value_is_rejected(bad) -> None:
    with pytest.raises(ValueError, match="kv_cache"):
        kavier.inference.performance({**ROW_A, "kv_cache": bad})


@pytest.mark.parametrize("flag", [True, False])
def test_kv_cache_accepts_numpy_bools(flag) -> None:
    # A dict row keeps the numpy type; a DataFrame row would be converted to a Python bool first.
    as_numpy = kavier.inference.performance({**ROW_A, "kv_cache": np.bool_(flag)})
    as_python = kavier.inference.performance({**ROW_A, "kv_cache": flag})
    assert as_numpy["total_s"].iloc[0] == as_python["total_s"].iloc[0]


def test_zero_gpu_hour_price_gives_zero_cost() -> None:
    out = kavier.inference.efficiency({**ROW_A, "gpu_hour_price": 0.0}).iloc[0]
    assert out["financial_per_mtoken"] == 0.0


@pytest.mark.parametrize(
    "verb",
    [kavier.inference.performance, kavier.inference.energy, kavier.inference.efficiency, kavier.inference.carbon],
)
def test_dataframe_index_is_kept_in_the_output(verb) -> None:
    batch = pd.DataFrame([ROW_A, ROW_B], index=["a", "b"])
    out = verb(batch)
    assert list(out.index) == ["a", "b"]
    # Assigning a predicted column back to the batch lines up row by row.
    col = out.columns[-1]
    batch[col] = out[col]
    assert batch[col].tolist() == out[col].tolist()


def test_identical_calls_give_identical_tasks() -> None:
    params = {**ROW_A, "kv_cache": True, "prefix_policy": "none", "prefix_min_tokens": 1024}
    first = run_inference(params)["_tasks"]
    second = run_inference(params)["_tasks"]
    assert first == second
    # 2026-01-01 00:00 UTC in ms since the Unix epoch.
    assert {t["submission_time"] for t in first} == {1_767_225_600_000}
