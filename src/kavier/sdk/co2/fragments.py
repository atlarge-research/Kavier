"""Build the carbon model's ``Fragment`` list from a Kavier training sim or an OpenDC powerSource parquet."""

from __future__ import annotations

from typing import List

import pandas as pd

from kavier.sdk.co2.engine import Fragment
from kavier.sdk.training.core.engine import _resolve_total_tokens, simulate_full_training, simulate_training_step


def fragments_from_training(
    *,
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    number_gpus: int,
    number_nodes: int,
    total_tokens: int | None,
    start_time: pd.Timestamp,
    epochs: float | None = None,
    dataset_tokens: int | None = None,
) -> List[Fragment]:
    """One Fragment: train_runtime (s) at aggregate power; sizing via _resolve_total_tokens (= kavier training)."""
    if _resolve_total_tokens(total_tokens, epochs, dataset_tokens) is None:
        raise ValueError("--total_tokens (or --epochs + --dataset_tokens) is required to derive a training runtime")

    total_gpus = number_gpus * number_nodes
    step = simulate_training_step(
        model_name=model_name,
        gpu_model=gpu_model,
        tokens_per_sample=tokens_per_sample,
        batch_size=batch_size,
        method=method,
        num_gpus=total_gpus,
        num_nodes=number_nodes,
    )
    summary = simulate_full_training(
        model_name=model_name,
        method=method,
        gpu_model=gpu_model,
        tokens_per_sample=tokens_per_sample,
        batch_size=batch_size,
        number_gpus=number_gpus,
        number_nodes=number_nodes,
        total_tokens=total_tokens,
        epochs=epochs,
        dataset_tokens=dataset_tokens,
    )
    per_gpu_power_w = step["gpu_power_watts"]
    aggregate_power_w = per_gpu_power_w * total_gpus
    runtime_s = summary["train_runtime"]
    return [Fragment(start_time=pd.Timestamp(start_time), duration_s=runtime_s, power_w=aggregate_power_w)]


def fragments_from_powersource(df: pd.DataFrame) -> List[Fragment]:
    """Per-timestamp Fragment: energy summed per ts, duration = gap to next distinct ts (last reuses prior width)."""
    if "timestamp" not in df.columns:
        raise ValueError(
            "powerSource parquet has no 'timestamp' column, so its energy cannot be "
            "placed on the carbon timeline; this input mode is unsupported."
        )
    if "energy_usage" not in df.columns:
        raise ValueError("powerSource parquet must contain an 'energy_usage' column (watt-seconds)")

    if pd.DatetimeIndex(df["timestamp"]).tz is not None:
        raise ValueError("powerSource timestamps must be timezone-naive")
    if len(df) == 0:
        return []

    # Duplicate timestamps (e.g. several sources sampled together) carry real energy:
    # sum energy per timestamp before diffing so zero-width rows are merged, not dropped.
    per_ts = df.groupby("timestamp", sort=True)["energy_usage"].sum()
    ts = pd.DatetimeIndex(per_ts.index)
    energy = per_ts.to_numpy()
    if len(ts) < 2:
        raise ValueError(
            "powerSource parquet has a single distinct timestamp, so row duration (and thus "
            "power) cannot be inferred; provide at least two distinct timestamps."
        )

    deltas = ts[1:] - ts[:-1]
    durations_s = [d.total_seconds() for d in deltas]
    durations_s.append(durations_s[-1])  # last row reuses the prior width

    frags: List[Fragment] = []
    for i in range(len(ts)):
        dur = durations_s[i]
        power_w = float(energy[i]) / dur
        frags.append(Fragment(start_time=ts[i], duration_s=dur, power_w=power_w))
    return frags
