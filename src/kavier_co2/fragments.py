"""Build power fragments from Kavier's training simulation or from an OpenDC
``powerSource.parquet``."""

from __future__ import annotations

from typing import List

import pandas as pd

from kavier_co2.emissions import Fragment
from kavier_training.core.engine import simulate_full_training, simulate_training_step


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
) -> List[Fragment]:
    """Simulate a training run and express it as one constant-power fragment.

    ``simulate_training_step`` gives the steady-state per-GPU power; aggregate
    power is that times the total GPU count. ``simulate_full_training`` gives the
    wall-clock runtime, which becomes the fragment duration.
    """
    if total_tokens is None:
        raise ValueError("--total_tokens is required to derive a training runtime")

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
    )
    per_gpu_power_w = step["gpu_power_watts"]
    aggregate_power_w = per_gpu_power_w * total_gpus
    runtime_s = summary["train_runtime"]
    return [Fragment(start_time=pd.Timestamp(start_time), duration_s=runtime_s, power_w=aggregate_power_w)]


def fragments_from_powersource(df: pd.DataFrame) -> List[Fragment]:
    """Convert an OpenDC ``powerSource.parquet`` into power fragments.

    OpenDC writes one row per sampling step with ``energy_usage`` in watt-seconds
    (Joules) consumed during that step. We require a ``timestamp`` column so each
    row can be placed on the carbon timeline; each row's duration is the gap to
    the next timestamp (the last row reuses the previous gap). Average power over
    a step is ``energy_usage / duration_s``.
    """
    if "timestamp" not in df.columns:
        raise ValueError(
            "powerSource parquet has no 'timestamp' column, so its energy cannot be "
            "placed on the carbon timeline; this input mode is unsupported."
        )
    if "energy_usage" not in df.columns:
        raise ValueError("powerSource parquet must contain an 'energy_usage' column (watt-seconds)")

    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(df["timestamp"])
    if ts.tz is not None:
        raise ValueError("powerSource timestamps must be timezone-naive")
    if len(df) == 0:
        return []

    deltas = ts[1:] - ts[:-1]  # TimedeltaIndex, unit-agnostic
    durations_s = [d.total_seconds() for d in deltas]
    if durations_s:
        durations_s.append(durations_s[-1])  # last row: reuse previous step width
    else:
        durations_s = [0.0]

    frags: List[Fragment] = []
    energy = df["energy_usage"].to_numpy()
    for i in range(len(df)):
        dur = durations_s[i]
        if dur <= 0:
            continue
        power_w = float(energy[i]) / dur
        frags.append(Fragment(start_time=ts[i], duration_s=dur, power_w=power_w))
    return frags
