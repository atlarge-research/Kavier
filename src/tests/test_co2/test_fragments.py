"""Tests for building power fragments from a training simulation and from an
OpenDC powerSource parquet."""

from __future__ import annotations

import pandas as pd
import pytest

from kavier_co2.fragments import (
    fragments_from_powersource,
    fragments_from_training,
)


def test_fragments_from_training_constant_power() -> None:
    """A training run yields constant-power fragments over its runtime."""
    frags = fragments_from_training(
        model_name="mistral-7b-v0.1",
        method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=8,
        number_nodes=1,
        total_tokens=1_000_000,
        start_time=pd.Timestamp("2025-06-01 00:00"),
    )
    assert len(frags) == 1
    f = frags[0]
    assert f.start_time == pd.Timestamp("2025-06-01 00:00")
    assert f.duration_s > 0
    # Power = per-GPU power * total GPUs, must be positive.
    assert f.power_w > 0


def test_fragments_from_training_power_scales_with_gpus() -> None:
    common = dict(
        model_name="mistral-7b-v0.1",
        method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_nodes=1,
        total_tokens=1_000_000,
        start_time=pd.Timestamp("2025-06-01 00:00"),
    )
    f1 = fragments_from_training(number_gpus=1, **common)[0]
    f8 = fragments_from_training(number_gpus=8, **common)[0]
    # 8 GPUs draw ~8x the aggregate power of 1 GPU at the same per-GPU power.
    assert f8.power_w == pytest.approx(8 * f1.power_w, rel=0.01)


def test_fragments_from_training_requires_total_tokens() -> None:
    with pytest.raises(ValueError):
        fragments_from_training(
            model_name="mistral-7b-v0.1",
            method="lora",
            gpu_model="NVIDIA-A100-SXM4-80GB",
            tokens_per_sample=1024,
            batch_size=4,
            number_gpus=8,
            number_nodes=1,
            total_tokens=None,
            start_time=pd.Timestamp("2025-06-01 00:00"),
        )


def test_fragments_from_powersource_with_timestamp() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 00:00:30", "2025-01-01 00:01:00"]
            ),
            "energy_usage": [3600.0, 7200.0, 3600.0],  # Ws per row
        }
    )
    frags = fragments_from_powersource(df)
    assert len(frags) == 3
    # Row 0 lasts 30 s (to next ts); 3600 Ws / 30 s = 120 W.
    assert frags[0].duration_s == pytest.approx(30.0)
    assert frags[0].power_w == pytest.approx(120.0)


def test_fragments_from_powersource_without_timestamp_raises() -> None:
    df = pd.DataFrame({"energy_usage": [3600.0, 7200.0]})
    with pytest.raises(ValueError):
        fragments_from_powersource(df)
