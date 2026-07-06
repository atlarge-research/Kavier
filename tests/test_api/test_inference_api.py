"""``kavier.inference`` batch API: shape contract, input-form equivalence, and numbers checked
against independent analytic oracles (never against the facade's own engine output).

The facade chains a single-stream roofline sim (``run_inference``) into flat-trace carbon/energy/cost.
For the numeric tests we take ``total_s`` from the ``performance`` verb (a plain input to the chain) and
re-derive energy/carbon/cost with the *closed-form* result of a constant-intensity integral — a
different arithmetic path than the ``Fragment``/``compute_emissions`` code under test.
"""

from __future__ import annotations

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

# Independent of the engine: n identical requests each emit (in + out) tokens.
TOKENS_A = ROW_A["num_requests"] * (ROW_A["input_tokens"] + ROW_A["output_tokens"])  # 16*160 = 2560


def _batch() -> pd.DataFrame:
    return pd.DataFrame([ROW_A, ROW_B])


@pytest.fixture(scope="module")
def perf_a() -> "pd.Series":
    """The ``performance`` row for ROW_A — its ``total_s`` seeds the energy/carbon/cost oracles below."""
    return kavier.inference.performance(ROW_A).iloc[0]


# ---------------------------------------------------------------------------------------------------
# Shape / input-form contract
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
    # Falsifies: a verb that collapses/reorders rows, drops the input columns, or omits a promised column.
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
    # Falsifies: a _normalise branch (DataFrame vs list vs dict) that produces a different workload.
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
    # Falsifies: dropping the _infer_params guards (would silently run or divide-by-zero instead of raising).
    with pytest.raises(ValueError):
        kavier.inference.performance(bad)


# ---------------------------------------------------------------------------------------------------
# performance: single-stream invariants (no independent physics oracle here — that's test_inference's job)
# ---------------------------------------------------------------------------------------------------


def test_throughput_is_total_tokens_over_total_time(perf_a) -> None:
    # Definitional invariant: throughput := total_tokens / total_s. Falsifies a per-request or
    # tokens/sum-of-latencies mixup, or dropping the ttft from total_s.
    assert perf_a["throughput_tok_s"] == pytest.approx(perf_a["total_tokens"] / perf_a["total_s"])
    assert perf_a["total_tokens"] == TOKENS_A  # engine's token count matches the hand count


def test_identical_requests_give_equal_percentiles_and_evenly_split_latency(perf_a) -> None:
    # Single-stream, no contention: 16 identical requests => identical per-request latency, so every
    # percentile collapses to the same value, which is the total wall time split n ways.
    # p50_ms oracle = total_s / num_requests * 1000. Falsifies a percentile/latency-units bug.
    assert perf_a["p50_ms"] == pytest.approx(perf_a["p95_ms"])
    assert perf_a["p50_ms"] == pytest.approx(perf_a["total_s"] / ROW_A["num_requests"] * 1000.0)


def test_mean_ttft_is_the_prefill_only_share_below_full_latency(perf_a) -> None:
    # TTFT is prefill time only; full latency adds the 32 decode tokens, so 0 < ttft < p50.
    # Falsifies a bug that reports the whole request latency (or 0) as time-to-first-token.
    assert 0.0 < perf_a["mean_ttft_ms"] < perf_a["p50_ms"]


# ---------------------------------------------------------------------------------------------------
# energy / carbon / cost: closed-form constant-intensity oracles
# ---------------------------------------------------------------------------------------------------


def test_energy_bills_gpu_max_power_over_busy_time(perf_a) -> None:
    total_s = perf_a["total_s"]
    max_power_w = get_gpu(ROW_A["gpu"]).max_power_w  # 150 W for the A10
    # Constant power P over time t => Wh = P*t/3600. Independent of the Fragment integrator.
    # Falsifies billing idle power, a /1000 vs /3600 slip, or an energy-per-token mixup.
    expected_wh = max_power_w * total_s / 3600.0
    out = kavier.inference.energy(ROW_A).iloc[0]
    assert out["energy_wh"] == pytest.approx(expected_wh)
    assert out["energy_per_mtoken_wh"] == pytest.approx(expected_wh * 1_000_000.0 / TOKENS_A)


def test_carbon_is_energy_kwh_times_grid_intensity(perf_a) -> None:
    total_s = perf_a["total_s"]
    max_power_w = get_gpu(ROW_A["gpu"]).max_power_w
    # A flat trace has no down-estimation, so gCO2 = (P*t / 3.6e6 kWh) * intensity. DEFAULT is 400 g/kWh.
    # Falsifies a wrong Ws->kWh constant, or billing against the wrong intensity.
    expected_co2_g = max_power_w * total_s / 3.6e6 * DEFAULT_INTENSITY_G_KWH
    out = kavier.inference.carbon(ROW_A).iloc[0]
    assert out["total_co2_g"] == pytest.approx(expected_co2_g)
    assert out["carbon_per_mtoken_g"] == pytest.approx(expected_co2_g * 1_000_000.0 / TOKENS_A)


def test_carbon_scales_linearly_with_the_intensity_column() -> None:
    # Doubling grid intensity doubles emissions; the `intensity` column must override the 400 default.
    # Falsifies an ignored intensity column or a non-linear/additive carbon model.
    base = kavier.inference.carbon({**ROW_A, "intensity": 400.0}).iloc[0]["total_co2_g"]
    doubled = kavier.inference.carbon({**ROW_A, "intensity": 800.0}).iloc[0]["total_co2_g"]
    assert doubled == pytest.approx(2.0 * base)


def test_cost_is_gpu_hours_times_price_with_column_override(perf_a) -> None:
    gpu_hours = perf_a["total_s"] / 3600.0
    # $/Mtoken = gpu_hours * $/hr * 1e6/tokens. Default price is 2.5; a column must override it.
    # Falsifies a wrong seconds->hours factor or an ignored gpu_hour_price column.
    expected_default = gpu_hours * DEFAULT_GPU_HOUR_PRICE * 1_000_000.0 / TOKENS_A
    out_default = kavier.inference.efficiency(ROW_A).iloc[0]
    assert out_default["financial_per_mtoken"] == pytest.approx(expected_default)

    out_priced = kavier.inference.efficiency({**ROW_A, "gpu_hour_price": 5.0}).iloc[0]
    assert out_priced["financial_per_mtoken"] == pytest.approx(gpu_hours * 5.0 * 1_000_000.0 / TOKENS_A)


# ---------------------------------------------------------------------------------------------------
# Facade default parameters
# ---------------------------------------------------------------------------------------------------


def test_facade_defaults_to_kv_cache_on(perf_a) -> None:
    # The facade must default kv_cache=True (linear decode), not False (quadratic). Pin the default by
    # showing the facade matches an explicit kv_cache=True run and diverges from kv_cache=False.
    # Falsifies flipping DEFAULT_KV_CACHE, or a default that skips the cache config entirely.
    base = {**ROW_A, "prefix_policy": "none", "prefix_min_tokens": 1024}
    on = run_inference({**base, "kv_cache": True})
    off = run_inference({**base, "kv_cache": False})
    assert perf_a["total_s"] == pytest.approx(on["total_s"])
    assert off["total_s"] > on["total_s"]  # quadratic decode without KV cache is strictly slower
