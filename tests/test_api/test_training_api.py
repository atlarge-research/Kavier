"""Tests for the ``kavier.training`` batch API.

Numeric checks use scaling laws (runtime linear in tokens), relations between columns
(samples/s = tokens/s / seq_len), an energy identity (E = P*t against the carbon integration),
bounds from the GPU spec (idle <= power <= max), and edge cases (unsized job -> runtime 0).
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


# --------------------------------------------------------------------------- output shape


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
    # One row per input row in order, input columns kept, documented columns present.
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
    # A dict, a list and a DataFrame of the same workload give identical rows.
    from_df = verb(pd.DataFrame([ROW_A])).iloc[0]
    from_list = verb([ROW_A]).iloc[0]
    from_dict = verb(ROW_A).iloc[0]
    pd.testing.assert_series_equal(from_df, from_list, check_names=False)
    pd.testing.assert_series_equal(from_df, from_dict, check_names=False)


# --------------------------------------------------------------------------- performance physics


def test_runtime_is_linear_in_total_tokens_at_fixed_throughput() -> None:
    # Throughput does not depend on job size and runtime = total_tokens / throughput, so
    # doubling tokens doubles runtime at equal tokens/s.
    small = _one(kavier.training.performance, {**ROW_A, "total_tokens": 10_000_000})
    large = _one(kavier.training.performance, {**ROW_A, "total_tokens": 20_000_000})
    assert large["train_tokens_per_second"] == pytest.approx(small["train_tokens_per_second"])
    assert large["train_runtime"] == pytest.approx(2.0 * small["train_runtime"])


def test_samples_per_second_is_tokens_per_second_over_seq_len() -> None:
    # One sample = seq_len tokens, so samples/s = (tokens/s) / seq_len.
    perf = _one(kavier.training.performance, ROW_A)
    assert perf["train_samples_per_second"] == pytest.approx(perf["train_tokens_per_second"] / ROW_A["seq_len"])


def test_gpu_power_within_idle_max_and_utilizations_are_percentages() -> None:
    # mse_power lies in [idle, max] of the GPU spec; utilizations are percentages in [0, 100].
    gpu = get_gpu(ROW_A["gpu"])
    perf = _one(kavier.training.performance, ROW_A)
    assert gpu.idle_power_w <= perf["gpu_power_watts"] <= gpu.max_power_w
    assert 0.0 <= perf["gpu_compute_utilization"] <= 100.0
    assert 0.0 <= perf["gpu_memory_utilization"] <= 100.0


# --------------------------------------------------------------------------- energy / carbon


def test_energy_wh_equals_aggregate_power_times_runtime() -> None:
    # Energy comes from integrating one power fragment over a flat carbon trace; check it
    # against E = P*t: Wh = aggregate_power_w * train_runtime / 3600. Fleet power = per-GPU
    # power x total_gpus (8); performance and energy compute these separately.
    perf = _one(kavier.training.performance, ROW_A)
    en = _one(kavier.training.energy, ROW_A)
    assert en["aggregate_power_w"] == pytest.approx(perf["gpu_power_watts"] * TOTAL_GPUS_A)
    expected_wh = en["aggregate_power_w"] * perf["train_runtime"] / 3600.0
    assert en["energy_wh"] == pytest.approx(expected_wh, rel=1e-6)


def test_carbon_and_energy_verbs_share_energy_and_default_intensity() -> None:
    # carbon and energy report equal energy for the same run, and carbon's default intensity
    # equals the exported DEFAULT_INTENSITY_G_KWH.
    en = _one(kavier.training.energy, ROW_A)
    ca_default = _one(kavier.training.carbon, ROW_A)
    ca_explicit = _one(kavier.training.carbon, {**ROW_A, "intensity": DEFAULT_INTENSITY_G_KWH})
    assert ca_default["total_energy_kwh"] == pytest.approx(en["energy_kwh"])
    assert ca_default["total_co2_g"] == pytest.approx(ca_explicit["total_co2_g"])


def test_co2_is_energy_times_intensity_and_scales_linearly() -> None:
    # Flat-trace carbon = energy [kWh] * intensity [g/kWh]; energy does not depend on intensity.
    # Doubling intensity doubles co2 and leaves energy unchanged.
    lo = _one(kavier.training.carbon, {**ROW_A, "intensity": 300.0})
    hi = _one(kavier.training.carbon, {**ROW_A, "intensity": 600.0})
    # Every trace window has the same intensity.
    assert lo["total_co2_g"] == pytest.approx(lo["total_energy_kwh"] * 300.0)
    assert hi["total_energy_kwh"] == pytest.approx(lo["total_energy_kwh"])
    assert hi["total_co2_g"] == pytest.approx(2.0 * lo["total_co2_g"])


def test_carbon_per_mtoken_is_total_over_million_tokens() -> None:
    # Per-Mtoken = total * 1e6 / total_tokens. For a 10 M-token job the factor is 0.1.
    ca = _one(kavier.training.carbon, ROW_A)
    assert ca["total_tokens"] == ROW_A["total_tokens"]
    per_m = 1_000_000.0 / ROW_A["total_tokens"]  # = 0.1
    assert ca["carbon_per_mtoken_g"] == pytest.approx(ca["total_co2_g"] * per_m)


# --------------------------------------------------------------------------- cost


def test_financial_per_mtoken_from_gpu_hours_and_price() -> None:
    # $/Mtoken = gpu_hours * $/hour * 1e6 / total_tokens, gpu_hours = runtime_h * total_gpus.
    price = 4.0
    perf = _one(kavier.training.performance, ROW_A)
    ef = _one(kavier.training.efficiency, {**ROW_A, "gpu_hour_price": price})
    expected_gpu_hours = perf["train_runtime"] / 3600.0 * TOTAL_GPUS_A
    assert ef["gpu_hours"] == pytest.approx(expected_gpu_hours)
    expected = expected_gpu_hours * price * 1_000_000.0 / ROW_A["total_tokens"]
    assert ef["financial_per_mtoken"] == pytest.approx(expected)


def test_cost_scales_linearly_with_price_and_default_is_the_constant() -> None:
    # $/Mtoken is linear in the hourly rate; no price means DEFAULT_GPU_HOUR_PRICE.
    base = _one(kavier.training.efficiency, {**ROW_A, "gpu_hour_price": 1.0})
    triple = _one(kavier.training.efficiency, {**ROW_A, "gpu_hour_price": 3.0})
    default = _one(kavier.training.efficiency, ROW_A)
    assert triple["financial_per_mtoken"] == pytest.approx(3.0 * base["financial_per_mtoken"])
    assert default["financial_per_mtoken"] == pytest.approx(base["financial_per_mtoken"] * DEFAULT_GPU_HOUR_PRICE)


# --------------------------------------------------------------------------- job-size resolution & edges


def test_epochs_times_dataset_tokens_sets_total_tokens() -> None:
    # Job size as epochs x dataset_tokens: 2 * 5 M = 10 M, same runtime as total_tokens = 10 M.
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
    # An omitted num_nodes equals num_nodes=1.
    no_nodes = {k: v for k, v in ROW_A.items() if k != "num_nodes"}
    omitted = _one(kavier.training.performance, no_nodes)
    explicit = _one(kavier.training.performance, {**no_nodes, "num_nodes": 1})
    assert omitted["train_runtime"] == pytest.approx(explicit["train_runtime"])
    assert omitted["gpu_power_watts"] == pytest.approx(explicit["gpu_power_watts"])


def test_unsized_job_has_zero_runtime_and_undefined_cost() -> None:
    # Without total_tokens or epochs: runtime 0, total_tokens None, $/Mtoken None.
    unsized = {k: v for k, v in ROW_A.items() if k != "total_tokens"}
    perf = _one(kavier.training.performance, unsized)
    assert perf["train_runtime"] == 0.0
    assert perf["total_tokens"] is None
    assert _one(kavier.training.efficiency, unsized)["financial_per_mtoken"] is None


@pytest.mark.parametrize("verb", [kavier.training.carbon, kavier.training.energy])
def test_billing_an_unsized_job_is_rejected(verb) -> None:
    # carbon and energy bill power over runtime; an unsized job has runtime 0 and raises.
    unsized = {k: v for k, v in ROW_A.items() if k != "total_tokens"}
    with pytest.raises(ValueError, match="runtime is 0"):
        verb(unsized)


@pytest.mark.parametrize(
    "size",
    [{"total_tokens": 0}, {"epochs": 0, "dataset_tokens": 500_000}, {"epochs": 2, "dataset_tokens": 0}],
)
def test_zero_job_size_is_kept_as_zero_tokens(size) -> None:
    # A zero cell used to read as missing: total_tokens came back None, and a zero epochs or
    # dataset_tokens raised "pass epochs and dataset_tokens together".
    row = {k: v for k, v in ROW_A.items() if k != "total_tokens"}
    perf = _one(kavier.training.performance, {**row, **size})
    assert perf["total_tokens"] == 0
    assert perf["train_runtime"] == 0.0


def test_billing_a_zero_token_job_names_the_job_size() -> None:
    with pytest.raises(ValueError, match="training runtime is 0; set a job size"):
        kavier.training.carbon({**ROW_A, "total_tokens": 0})


@pytest.mark.parametrize(
    "verb",
    [kavier.training.performance, kavier.training.energy, kavier.training.efficiency, kavier.training.carbon],
)
def test_dataframe_index_is_kept_in_the_output(verb) -> None:
    batch = pd.DataFrame([ROW_A, ROW_B], index=["a", "b"])
    out = verb(batch)
    assert list(out.index) == ["a", "b"]
