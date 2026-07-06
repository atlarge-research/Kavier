"""``kavier.training`` batch API.

Every numeric assertion here is anchored to an oracle that is *independent* of the batch
facade under test: a scaling law (runtime linear in tokens), a cross-column algebraic
relation the facade never asserts itself (samples/s = tokens/s ÷ seq_len), a cross-method
energy identity (E = P·t vs. the carbon-integration path), a physical bound read from the
GPU spec (power ∈ [idle, max]), or a hand-evaluated edge case (unsized job → runtime 0).
None of them paste an engine output back into ``assert ==``.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kavier
from kavier.sdk.inference.facade import DEFAULT_GPU_HOUR_PRICE
from kavier.sdk.library.lookup import get_gpu
from kavier.sdk.training import DEFAULT_INTENSITY_G_KWH

# num_gpus=8, num_nodes=1  ->  total_gpus = 8. seq_len=1024. total_tokens = 10 M.
ROW_A = {
    "model": "mistral-7b-v0.1",
    "gpu": "NVIDIA-A100-SXM4-80GB",
    "method": "lora",
    "batch_size": 4,
    "seq_len": 1024,
    "num_gpus": 8,
    "num_nodes": 1,
    "total_tokens": 10_000_000,
}
ROW_B = {
    "model": "mistral-7b-v0.1",
    "gpu": "NVIDIA-A100-SXM4-80GB",
    "method": "full",
    "batch_size": 2,
    "seq_len": 2048,
    "num_gpus": 4,
    "num_nodes": 1,
    "total_tokens": 20_000_000,
}
TOTAL_GPUS_A = ROW_A["num_gpus"] * ROW_A["num_nodes"]  # 8


def _one(verb, row):
    return verb(row).iloc[0]


def _batch() -> pd.DataFrame:
    return pd.DataFrame([ROW_A, ROW_B])


# --------------------------------------------------------------------------- shape contract


@pytest.mark.parametrize(
    ("verb", "expected_cols"),
    [
        (
            kavier.training.performance,
            ("train_tokens_per_second", "train_runtime", "gpu_compute_utilization", "gpu_power_watts"),
        ),
        (kavier.training.energy, ("energy_wh", "energy_per_mtoken_wh")),
        (kavier.training.efficiency, ("financial_per_mtoken",)),
        (kavier.training.carbon, ("carbon_per_mtoken_g", "total_co2_g")),
    ],
)
def test_verb_emits_predicted_columns_and_preserves_every_input_row(verb, expected_cols) -> None:
    # Contract: one output row per input row, input columns carried through unchanged, and
    # each verb's documented predicted columns present. Falsifies if a verb drops/reorders
    # rows, silently swallows an input column, or stops emitting a documented output column.
    batch = _batch()
    out = verb(batch)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(batch)
    assert list(out["method"]) == [ROW_A["method"], ROW_B["method"]]
    assert list(out["batch_size"]) == [ROW_A["batch_size"], ROW_B["batch_size"]]
    for col in expected_cols:
        assert col in out.columns


@pytest.mark.parametrize(
    "verb",
    [kavier.training.performance, kavier.training.energy, kavier.training.efficiency, kavier.training.carbon],
)
def test_single_dict_list_and_dataframe_produce_identical_rows(verb) -> None:
    # Invariant: the three accepted input forms are just different containers for the same
    # workload, so they must yield bit-identical predictions. Falsifies if _normalise treats
    # a dict/list/DataFrame row differently (e.g. dtype coercion or a dropped key).
    from_df = verb(pd.DataFrame([ROW_A])).iloc[0]
    from_list = verb([ROW_A]).iloc[0]
    from_dict = verb(ROW_A).iloc[0]
    pd.testing.assert_series_equal(from_df, from_list, check_names=False)
    pd.testing.assert_series_equal(from_df, from_dict, check_names=False)


# --------------------------------------------------------------------------- performance physics


def test_runtime_is_linear_in_total_tokens_at_fixed_throughput() -> None:
    # Scaling law: throughput is a per-step property independent of job size, and
    # runtime = total_tokens / throughput. So doubling the token budget must double the
    # runtime while leaving tokens/s untouched. Falsifies if runtime stops scaling with
    # tokens or throughput leaks a dependence on total_tokens.
    small = _one(kavier.training.performance, {**ROW_A, "total_tokens": 10_000_000})
    large = _one(kavier.training.performance, {**ROW_A, "total_tokens": 20_000_000})
    assert large["train_tokens_per_second"] == pytest.approx(small["train_tokens_per_second"])
    assert large["train_runtime"] == pytest.approx(2.0 * small["train_runtime"])


def test_samples_per_second_is_tokens_per_second_over_seq_len() -> None:
    # Cross-column identity the facade never asserts itself: one sample = seq_len tokens,
    # so samples/s = (tokens/s) / seq_len. Falsifies if samples/s is derived from
    # batch_size or a different token count.
    perf = _one(kavier.training.performance, ROW_A)
    assert perf["train_samples_per_second"] == pytest.approx(perf["train_tokens_per_second"] / ROW_A["seq_len"])


def test_gpu_power_within_idle_max_and_utilizations_are_percentages() -> None:
    # Physical bounds from the GPU spec (not hard-coded): mse_power is confined to
    # [idle, max], and both utilizations are clamped fractions reported as percent [0, 100].
    # Falsifies if the power model returns an out-of-envelope watt figure or a util > 100.
    gpu = get_gpu(ROW_A["gpu"])
    perf = _one(kavier.training.performance, ROW_A)
    assert gpu.idle_power_w <= perf["gpu_power_watts"] <= gpu.max_power_w
    assert 0.0 <= perf["gpu_compute_utilization"] <= 100.0
    assert 0.0 <= perf["gpu_memory_utilization"] <= 100.0


# --------------------------------------------------------------------------- energy / carbon


def test_energy_wh_equals_aggregate_power_times_runtime() -> None:
    # Cross-method check: the reported energy comes from integrating one power fragment over a
    # flat carbon trace; an *independent* route is E = P·t. With aggregate_power_w watts held
    # for train_runtime seconds, Wh = W · s / 3600. This catches the classic /1000 vs /3600
    # unit bug in the kWh<->Wh conversion. Also pins the aggregation: fleet power = per-GPU
    # power × total_gpus (8), which performance and energy compute on separate paths.
    perf = _one(kavier.training.performance, ROW_A)
    en = _one(kavier.training.energy, ROW_A)
    assert en["aggregate_power_w"] == pytest.approx(perf["gpu_power_watts"] * TOTAL_GPUS_A)
    expected_wh = en["aggregate_power_w"] * perf["train_runtime"] / 3600.0
    assert en["energy_wh"] == pytest.approx(expected_wh, rel=1e-6)


def test_carbon_and_energy_verbs_share_energy_and_default_intensity() -> None:
    # The carbon and energy verbs must report the SAME energy for the same run (they share
    # the billing path), and carbon's default intensity must be the exported constant, not a
    # private literal. Falsifies if the two verbs diverge on energy or the default drifts.
    en = _one(kavier.training.energy, ROW_A)
    ca_default = _one(kavier.training.carbon, ROW_A)
    ca_explicit = _one(kavier.training.carbon, {**ROW_A, "intensity": DEFAULT_INTENSITY_G_KWH})
    assert ca_default["total_energy_kwh"] == pytest.approx(en["energy_kwh"])
    assert ca_default["total_co2_g"] == pytest.approx(ca_explicit["total_co2_g"])


def test_co2_is_energy_times_intensity_and_scales_linearly() -> None:
    # Flat-trace carbon = energy(kWh) · intensity(g/kWh), and energy is independent of
    # intensity. So (a) at an explicit intensity, co2_g = energy_kwh · intensity exactly, and
    # (b) doubling intensity doubles co2 while energy is unchanged. Falsifies if intensity is
    # mis-applied (e.g. multiplied into energy) or the flat trace down-estimates unequally.
    lo = _one(kavier.training.carbon, {**ROW_A, "intensity": 300.0})
    hi = _one(kavier.training.carbon, {**ROW_A, "intensity": 600.0})
    # 300 g/kWh · energy_kwh  ==  co2_g  (all trace windows carry the same intensity).
    assert lo["total_co2_g"] == pytest.approx(lo["total_energy_kwh"] * 300.0)
    assert hi["total_energy_kwh"] == pytest.approx(lo["total_energy_kwh"])
    assert hi["total_co2_g"] == pytest.approx(2.0 * lo["total_co2_g"])


def test_carbon_per_mtoken_is_total_over_million_tokens() -> None:
    # Per-Mtoken = total · 1e6 / total_tokens. With a 10 M-token job the factor is 0.1.
    # Falsifies if the intensive metric uses the wrong divisor (e.g. /1000 instead of /1e6).
    ca = _one(kavier.training.carbon, ROW_A)
    assert ca["total_tokens"] == ROW_A["total_tokens"]
    per_m = 1_000_000.0 / ROW_A["total_tokens"]  # = 0.1
    assert ca["carbon_per_mtoken_g"] == pytest.approx(ca["total_co2_g"] * per_m)


# --------------------------------------------------------------------------- cost


def test_financial_per_mtoken_from_gpu_hours_and_price() -> None:
    # $/Mtoken = gpu_hours · $/hour · 1e6 / total_tokens, where gpu_hours = runtime_h ·
    # total_gpus. Cross-checked against performance's runtime and an EXPLICIT price so the
    # oracle owns every factor. Falsifies if cost forgets total_gpus or mis-bills the rate.
    price = 4.0
    perf = _one(kavier.training.performance, ROW_A)
    ef = _one(kavier.training.efficiency, {**ROW_A, "gpu_hour_price": price})
    expected_gpu_hours = perf["train_runtime"] / 3600.0 * TOTAL_GPUS_A
    assert ef["gpu_hours"] == pytest.approx(expected_gpu_hours)
    expected = expected_gpu_hours * price * 1_000_000.0 / ROW_A["total_tokens"]
    assert ef["financial_per_mtoken"] == pytest.approx(expected)


def test_cost_scales_linearly_with_price_and_default_is_the_constant() -> None:
    # $/Mtoken is linear in the GPU hourly rate, and omitting the price uses the exported
    # default. Falsifies if the rate enters non-linearly or the default literal drifts.
    base = _one(kavier.training.efficiency, {**ROW_A, "gpu_hour_price": 1.0})
    triple = _one(kavier.training.efficiency, {**ROW_A, "gpu_hour_price": 3.0})
    default = _one(kavier.training.efficiency, ROW_A)
    assert triple["financial_per_mtoken"] == pytest.approx(3.0 * base["financial_per_mtoken"])
    assert default["financial_per_mtoken"] == pytest.approx(base["financial_per_mtoken"] * DEFAULT_GPU_HOUR_PRICE)


# --------------------------------------------------------------------------- job-size resolution & edges


def test_epochs_times_dataset_tokens_sets_total_tokens() -> None:
    # Job size may be given as epochs × dataset_tokens instead of total_tokens:
    # 2 epochs · 5 M tokens = 10 M, which must then drive the same runtime as an explicit
    # 10 M budget. Falsifies if the epochs->tokens resolution is wrong or ignored.
    epoched = _one(
        kavier.training.performance,
        {
            "model": ROW_A["model"],
            "gpu": ROW_A["gpu"],
            "method": ROW_A["method"],
            "batch_size": ROW_A["batch_size"],
            "seq_len": ROW_A["seq_len"],
            "num_gpus": ROW_A["num_gpus"],
            "num_nodes": ROW_A["num_nodes"],
            "epochs": 2,
            "dataset_tokens": 5_000_000,
        },
    )
    explicit = _one(kavier.training.performance, ROW_A)  # total_tokens = 10 M
    assert epoched["total_tokens"] == 10_000_000
    assert epoched["train_runtime"] == pytest.approx(explicit["train_runtime"])


def test_num_nodes_defaults_to_one() -> None:
    # A batch commonly omits num_nodes; the default must be 1, so an omitted row equals an
    # explicit num_nodes=1 row. Falsifies if DEFAULT_NUM_NODES changes or the default is
    # applied inconsistently across the total_gpus math.
    no_nodes = {k: v for k, v in ROW_A.items() if k != "num_nodes"}
    omitted = _one(kavier.training.performance, no_nodes)
    explicit = _one(kavier.training.performance, {**no_nodes, "num_nodes": 1})
    assert omitted["train_runtime"] == pytest.approx(explicit["train_runtime"])
    assert omitted["gpu_power_watts"] == pytest.approx(explicit["gpu_power_watts"])


def test_unsized_job_has_zero_runtime_and_undefined_cost() -> None:
    # With no total_tokens / epochs the job has no size: runtime is 0, total_tokens is None,
    # and $/Mtoken is undefined (None), not a divide-by-zero. Falsifies if an unsized job
    # fabricates a runtime or emits a bogus finite cost.
    unsized = {k: v for k, v in ROW_A.items() if k != "total_tokens"}
    perf = _one(kavier.training.performance, unsized)
    assert perf["train_runtime"] == 0.0
    assert perf["total_tokens"] is None
    assert _one(kavier.training.efficiency, unsized)["financial_per_mtoken"] is None


@pytest.mark.parametrize("verb", [kavier.training.carbon, kavier.training.energy])
def test_billing_an_unsized_job_is_rejected(verb) -> None:
    # Carbon/energy bill power over runtime; a zero-runtime (unsized) job cannot be billed and
    # must raise rather than silently report zero emissions. Falsifies if the guard is removed
    # and an unsized job returns 0 gCO2 / 0 Wh.
    unsized = {k: v for k, v in ROW_A.items() if k != "total_tokens"}
    with pytest.raises(ValueError, match="runtime is 0"):
        verb(unsized)
