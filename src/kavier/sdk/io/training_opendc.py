"""Export a simulated training job to OpenDC task/fragment parquet input."""

from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd

from kavier.sdk.io.opendc.adapter import prepare_opendc_input
from kavier.sdk.library.lookup import get_gpu, get_llm


def estimate_task_memory_mb(llm_params: float, gpu_memory_gb: float, total_gpus: int) -> int:
    """Estimate per-task memory (MB): sharded FP16 params + 25% activation, min 1024."""
    parameter_memory_mb = (llm_params * 16.0) / (8.0 * 1024.0 * 1024.0 * max(1, total_gpus))
    activation_memory_mb = gpu_memory_gb * 1024.0 * 0.25
    return int(max(1024.0, parameter_memory_mb + activation_memory_mb))


def build_training_opendc_frames(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    number_gpus: int,
    number_nodes: int,
    total_tokens: int | None,
    task_id: int,
    submission_time_ms: int,
    simulate_full_training_fn: Callable[..., dict[str, Any]],
    simulate_training_step_fn: Callable[..., dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build OpenDC (tasks_df, fragments_df, summary) for one training job: one fragment per simulated step."""
    llm = get_llm(model_name)
    gpu = get_gpu(gpu_model)
    summary = simulate_full_training_fn(
        model_name=model_name,
        method=method,
        gpu_model=gpu_model,
        tokens_per_sample=tokens_per_sample,
        batch_size=batch_size,
        number_gpus=number_gpus,
        number_nodes=number_nodes,
        total_tokens=total_tokens,
    )
    total_gpus = number_gpus * number_nodes
    step_metrics = simulate_training_step_fn(
        model_name=model_name,
        gpu_model=gpu_model,
        tokens_per_sample=tokens_per_sample,
        batch_size=batch_size,
        method=method,
        num_gpus=total_gpus,
        num_nodes=number_nodes,
    )

    step_duration_ms = max(1, int(round(step_metrics["step_time_ms"])))
    if total_tokens is not None:
        estimated_runtime_ms = max(1, int(round(summary["train_runtime"] * 1000)))
        step_count = max(1, int(math.ceil(estimated_runtime_ms / step_duration_ms)))
        task_duration_ms = step_count * step_duration_ms
    else:
        step_count = 1
        task_duration_ms = step_duration_ms

    cpu_capacity = 1000.0
    mem_capacity = estimate_task_memory_mb(llm.m_params, gpu.memory_gb, total_gpus)
    gpu_capacity = float(gpu.core_max_mhz)

    tasks = pd.DataFrame(
        [
            {
                "id": task_id,
                "submission_time": submission_time_ms,
                "duration": task_duration_ms,
                "cpu_count": int(total_gpus),
                "cpu_capacity": cpu_capacity,
                "mem_capacity": mem_capacity,
                "gpu_count": int(total_gpus),
                "gpu_capacity": gpu_capacity,
            }
        ]
    )

    fragments_data: list[dict[str, object]] = [
        {
            "id": task_id,
            "duration": step_duration_ms,
            "cpu_count": int(total_gpus),
            "cpu_usage": cpu_capacity * total_gpus,
            "gpu_count": int(total_gpus),
            "gpu_usage": gpu_capacity * (step_metrics["gpu_compute_utilization"] / 100.0) * total_gpus,
        }
        for _ in range(step_count)
    ]

    fragments = pd.DataFrame(fragments_data)
    return tasks, fragments, summary


def export_training_opendc(
    output_dir: str,
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    number_gpus: int,
    number_nodes: int,
    total_tokens: int | None,
    task_id: int,
    submission_time_ms: int,
    simulate_full_training_fn: Callable[..., dict[str, Any]],
    simulate_training_step_fn: Callable[..., dict[str, float]],
) -> dict[str, Any]:
    """Build the OpenDC frames, write them to ``output_dir``, and return the full-training summary."""
    tasks, fragments, summary = build_training_opendc_frames(
        model_name=model_name,
        method=method,
        gpu_model=gpu_model,
        tokens_per_sample=tokens_per_sample,
        batch_size=batch_size,
        number_gpus=number_gpus,
        number_nodes=number_nodes,
        total_tokens=total_tokens,
        task_id=task_id,
        submission_time_ms=submission_time_ms,
        simulate_full_training_fn=simulate_full_training_fn,
        simulate_training_step_fn=simulate_training_step_fn,
    )
    prepare_opendc_input(tasks, fragments, output_dir)
    return summary
