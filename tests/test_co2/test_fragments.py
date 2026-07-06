"""Building power ``Fragment``s from a training simulation and from an OpenDC powerSource parquet.

Oracles here are independent of ``fragments.py``'s own arithmetic:
- powerSource fragments use hand-derived energy/duration/power values (P = E/Δt);
- training fragments are cross-checked against the underlying training engine it composes
  (``simulate_training_step`` / ``simulate_full_training``) and against GPU physical bounds.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kavier.sdk.co2.fragments import (
    fragments_from_powersource,
    fragments_from_training,
)
from kavier.sdk.library.lookup import get_gpu
from kavier.sdk.training.core.engine import (
    simulate_full_training,
    simulate_training_step,
)

_TRAIN = dict(
    model_name="mistral-7b-v0.1",
    method="lora",
    gpu_model="NVIDIA-A100-SXM4-80GB",
    tokens_per_sample=1024,
    batch_size=4,
    number_nodes=1,
    start_time=pd.Timestamp("2025-06-01 00:00"),
)


# --------------------------------------------------------------------------- #
# fragments_from_training
# --------------------------------------------------------------------------- #


def test_training_emits_one_fragment_preserving_start() -> None:
    # A whole job is modelled as a single constant-power interval; the start_time passes through unchanged.
    frags = fragments_from_training(number_gpus=8, total_tokens=1_000_000, **_TRAIN)
    assert len(frags) == 1  # falsify: emitting one fragment per step would make this >1
    assert frags[0].start_time == pd.Timestamp("2025-06-01 00:00")  # falsify: shifting/dropping start_time


def test_training_duration_is_whole_job_runtime_not_step_time() -> None:
    # Duration must be the full-job runtime (total_tokens / train_tps), not the millisecond step time.
    summary = simulate_full_training(
        model_name=_TRAIN["model_name"],
        method=_TRAIN["method"],
        gpu_model=_TRAIN["gpu_model"],
        tokens_per_sample=_TRAIN["tokens_per_sample"],
        batch_size=_TRAIN["batch_size"],
        number_gpus=8,
        number_nodes=1,
        total_tokens=1_000_000,
    )
    frag = fragments_from_training(number_gpus=8, total_tokens=1_000_000, **_TRAIN)[0]
    # Cross-check against the engine the builder wraps: runtime = total_tokens / train_tokens_per_second.
    # A bug substituting step_time_ms/1000 (~1.4 s) would diverge hugely from this ~30 s runtime.
    assert frag.duration_s == pytest.approx(summary["train_runtime"])  # falsify: using step_time_ms
    # Independent lower bound: 1e6 tokens can't be trained faster than one step, so runtime > 0 and
    # far exceeds a single ~1.4 s step -> a good sanity floor that a constant-0 duration would violate.
    assert frag.duration_s > 1.0


def test_training_power_is_per_gpu_times_total_gpus() -> None:
    # Fragment power is the AGGREGATE draw: per-GPU power (from the step engine) x total GPUs.
    number_gpus, number_nodes = 8, 1
    total_gpus = number_gpus * number_nodes
    step = simulate_training_step(
        model_name=_TRAIN["model_name"],
        gpu_model=_TRAIN["gpu_model"],
        tokens_per_sample=_TRAIN["tokens_per_sample"],
        batch_size=_TRAIN["batch_size"],
        method=_TRAIN["method"],
        num_gpus=total_gpus,
        num_nodes=number_nodes,
    )
    frag = fragments_from_training(number_gpus=number_gpus, total_tokens=1_000_000, **_TRAIN)[0]
    # falsify: dropping the "x total_gpus" (billing only one GPU) makes this off by 8x.
    assert frag.power_w == pytest.approx(step["gpu_power_watts"] * total_gpus)


def test_training_aggregate_power_within_physical_bounds() -> None:
    # Independent of the engine: N GPUs draw between N*idle and N*max watts (mse_power stays in [idle, max]).
    number_gpus = 8
    gpu = get_gpu(_TRAIN["gpu_model"])
    frag = fragments_from_training(number_gpus=number_gpus, total_tokens=1_000_000, **_TRAIN)[0]
    assert number_gpus * gpu.idle_power_w <= frag.power_w <= number_gpus * gpu.max_power_w
    # falsify: returning 0 W, or forgetting the aggregate multiply, breaks the lower bound.


def test_training_aggregate_power_scales_linearly_with_gpu_count() -> None:
    # Per-GPU draw here is compute-util-bound (independent of GPU count), so aggregate power is exactly linear.
    f1 = fragments_from_training(number_gpus=1, total_tokens=1_000_000, **_TRAIN)[0]
    f8 = fragments_from_training(number_gpus=8, total_tokens=1_000_000, **_TRAIN)[0]
    assert f8.power_w == pytest.approx(8.0 * f1.power_w, rel=1e-9)  # falsify: billing per-GPU -> ratio 1.0


def test_training_epochs_and_dataset_tokens_equal_explicit_total_tokens() -> None:
    # total_tokens = epochs * dataset_tokens (2 * 500_000 = 1_000_000), so the runtimes must match exactly.
    explicit = fragments_from_training(number_gpus=8, total_tokens=1_000_000, **_TRAIN)[0]
    derived = fragments_from_training(number_gpus=8, total_tokens=None, epochs=2.0, dataset_tokens=500_000, **_TRAIN)[0]
    # falsify: ignoring epochs/dataset_tokens (or mis-deriving total_tokens) diverges the two durations.
    assert derived.duration_s == pytest.approx(explicit.duration_s)


def test_training_requires_a_job_size() -> None:
    # No total_tokens and no epochs+dataset_tokens => no runtime can be derived.
    with pytest.raises(ValueError):
        fragments_from_training(number_gpus=8, total_tokens=None, **_TRAIN)


# --------------------------------------------------------------------------- #
# fragments_from_powersource
# --------------------------------------------------------------------------- #


def test_powersource_regular_grid_power_is_energy_over_gap() -> None:
    # Three samples 30 s apart; the last row reuses the prior 30 s width. P = energy / duration.
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:00:30", "2025-01-01 00:01:00"]),
            "energy_usage": [3600.0, 7200.0, 3600.0],  # watt-seconds per row
        }
    )
    frags = fragments_from_powersource(df)
    assert len(frags) == 3
    # Hand-derived (E / Δt): 3600/30=120, 7200/30=240, last reuses 30 s -> 3600/30=120.
    assert [f.duration_s for f in frags] == pytest.approx([30.0, 30.0, 30.0])
    assert [f.power_w for f in frags] == pytest.approx([120.0, 240.0, 120.0])
    assert frags[0].start_time == pd.Timestamp("2025-01-01 00:00:00")
    assert frags[2].start_time == pd.Timestamp("2025-01-01 00:01:00")


def test_powersource_duplicate_timestamps_conserve_energy() -> None:
    # Regression (issue #8): sources sampled at the same timestamps must have their energy summed, not dropped.
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 00:00:00", "2025-01-01 00:05:00", "2025-01-01 00:05:00"]
            ),
            "energy_usage": [2000.0, 2000.0, 3000.0, 3000.0],  # 10,000 Ws total
        }
    )
    frags = fragments_from_powersource(df)
    assert len(frags) == 2  # two DISTINCT timestamps
    # Total energy must be conserved: 2000+2000+3000+3000 = 10,000 Ws.
    total_ws = sum(f.power_w * f.duration_s for f in frags)
    assert total_ws == pytest.approx(10_000.0)  # falsify: dropping a duplicate row -> 8,000 Ws
    # 00:00 rows merge to 4,000 Ws over the 300 s gap; last row reuses the 300 s width for 6,000 Ws.
    assert frags[0].start_time == pd.Timestamp("2025-01-01 00:00:00")
    assert frags[0].duration_s == pytest.approx(300.0)
    assert frags[0].power_w == pytest.approx(4000.0 / 300.0)
    assert frags[1].duration_s == pytest.approx(300.0)
    assert frags[1].power_w == pytest.approx(6000.0 / 300.0)  # = 20.0 W


def test_powersource_missing_timestamp_column_raises() -> None:
    # Without timestamps the energy cannot be placed on the carbon timeline.
    with pytest.raises(ValueError, match="timestamp"):
        fragments_from_powersource(pd.DataFrame({"energy_usage": [3600.0, 7200.0]}))


def test_powersource_missing_energy_column_raises() -> None:
    # Timestamps but no energy -> nothing to integrate.
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:01"])})
    with pytest.raises(ValueError, match="energy_usage"):
        fragments_from_powersource(df)


def test_powersource_timezone_aware_timestamps_raise() -> None:
    # The carbon trace is timezone-naive; tz-aware powerSource timestamps are rejected to avoid silent offsets.
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:01"]).tz_localize("UTC"),
            "energy_usage": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="timezone-naive"):
        fragments_from_powersource(df)


def test_powersource_single_distinct_timestamp_raises() -> None:
    # Regression (issue #8): one timestamp gives no gap to infer duration -> must raise, not silently drop energy.
    single = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01 00:00:00"]), "energy_usage": [3600.0]})
    with pytest.raises(ValueError, match="distinct timestamp"):
        fragments_from_powersource(single)
    # Several rows all at ONE timestamp are just as undatable (they collapse to one group).
    dup_only = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2025-01-01 00:00:00"] * 2), "energy_usage": [3600.0, 3600.0]}
    )
    with pytest.raises(ValueError, match="distinct timestamp"):
        fragments_from_powersource(dup_only)


def test_powersource_empty_returns_no_fragments() -> None:
    df = pd.DataFrame({"timestamp": pd.to_datetime([]), "energy_usage": pd.Series([], dtype=float)})
    assert fragments_from_powersource(df) == []  # falsify: raising or fabricating a fragment
