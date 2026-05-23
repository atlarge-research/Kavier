from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from kavier_io.training_opendc import build_training_opendc_frames, export_training_opendc
from kavier_training.core.engine import simulate_full_training, simulate_training_step

TRAINING_CONFIGS = [
    {"model_name": "mistral-7b-v0.1", "gpu_model": "NVIDIA-A100-SXM4-80GB", "method": "full"},
    {"model_name": "mistral-7b-v0.1", "gpu_model": "NVIDIA-A100-80GB-PCIe", "method": "lora"},
    {"model_name": "llama3.1-70b", "gpu_model": "NVIDIA-A100-SXM4-80GB", "method": "lora"},
    {"model_name": "granite-3.1-3b-a800m-instruct", "gpu_model": "NVIDIA-A100-80GB-PCIe", "method": "full"},
    {"model_name": "mixtral-8x7b-instruct-v0.1", "gpu_model": "NVIDIA-A100-SXM4-80GB", "method": "full"},
    {"model_name": "mixtral-8x7b-instruct-v0.1", "gpu_model": "NVIDIA-A100-80GB-PCIe", "method": "gptq-lora"},
]


@pytest.mark.parametrize("cfg", TRAINING_CONFIGS, ids=[f"{c['model_name']}_{c['method']}" for c in TRAINING_CONFIGS])
def test_simulate_training_step_returns_valid_metrics(cfg):
    result = simulate_training_step(
        model_name=cfg["model_name"],
        gpu_model=cfg["gpu_model"],
        tokens_per_sample=1024,
        batch_size=4,
        method=cfg["method"],
        num_gpus=1,
    )

    assert isinstance(result, dict)
    assert result["tokens_per_second"] > 0
    assert result["step_time_ms"] > 0
    assert result["gpu_power_watts"] > 0
    assert 0 <= result["gpu_memory_utilization"] <= 100


def test_multi_gpu_increases_throughput():
    r1 = simulate_training_step(
        "mistral-7b-v0.1",
        "NVIDIA-A100-SXM4-80GB",
        1024,
        4,
        "full",
        num_gpus=1,
    )
    r4 = simulate_training_step(
        "mistral-7b-v0.1",
        "NVIDIA-A100-SXM4-80GB",
        1024,
        4,
        "full",
        num_gpus=4,
    )
    assert r4["tokens_per_second"] > r1["tokens_per_second"] * 0.5


def test_lora_step_not_slower_than_full():
    full = simulate_training_step(
        "mistral-7b-v0.1",
        "NVIDIA-A100-SXM4-80GB",
        1024,
        4,
        "full",
    )
    lora = simulate_training_step(
        "mistral-7b-v0.1",
        "NVIDIA-A100-SXM4-80GB",
        1024,
        4,
        "lora",
    )
    assert lora["step_time_ms"] <= full["step_time_ms"]


def test_unsupported_model_raises_key_error():
    with pytest.raises(KeyError):
        simulate_training_step(
            "nonexistent-model",
            "NVIDIA-A100-SXM4-80GB",
            1024,
            4,
            "full",
        )


def test_simulate_full_training_returns_complete_result():
    result = simulate_full_training(
        model_name="mistral-7b-v0.1",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=2,
        number_nodes=1,
        total_tokens=1_000_000,
    )

    assert result["train_tokens_per_second"] > 0
    assert result["train_runtime"] > 0
    assert result["number_gpus"] == 2
    assert result["model_name"] == "mistral-7b-v0.1"


def test_simulate_full_training_without_total_tokens():
    result = simulate_full_training(
        model_name="mistral-7b-v0.1",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=1,
        number_nodes=1,
    )

    assert result["train_tokens_per_second"] > 0
    assert result["train_runtime"] == 0.0


def test_build_opendc_frames_strict_columns_and_values():
    tasks, fragments, _summary = build_training_opendc_frames(
        model_name="mistral-7b-v0.1",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=2,
        number_nodes=1,
        total_tokens=1_000_000,
        task_id=1,
        submission_time_ms=1234,
        simulate_full_training_fn=simulate_full_training,
        simulate_training_step_fn=simulate_training_step,
    )

    assert list(tasks.columns) == [
        "id",
        "submission_time",
        "duration",
        "cpu_count",
        "cpu_capacity",
        "mem_capacity",
        "gpu_count",
        "gpu_capacity",
    ]
    assert list(fragments.columns) == [
        "id",
        "duration",
        "cpu_count",
        "cpu_usage",
        "gpu_count",
        "gpu_usage",
    ]
    assert tasks.loc[0, "id"] == 1
    assert tasks.loc[0, "submission_time"] == 1234
    assert tasks.loc[0, "duration"] > 0
    assert len(fragments) > 1
    assert fragments["id"].nunique() == 1
    assert fragments["duration"].sum() == tasks.loc[0, "duration"]


def test_export_training_opendc_writes_parquet_with_strict_schema(tmp_path: Path):
    result = export_training_opendc(
        output_dir=str(tmp_path),
        model_name="mistral-7b-v0.1",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=1,
        number_nodes=1,
        total_tokens=100_000,
        task_id=7,
        submission_time_ms=0,
        simulate_full_training_fn=simulate_full_training,
        simulate_training_step_fn=simulate_training_step,
    )

    assert result["train_tokens_per_second"] > 0

    tasks_table = pq.read_table(tmp_path / "tasks.parquet")
    fragments_table = pq.read_table(tmp_path / "fragments.parquet")

    assert tasks_table.column_names == [
        "id",
        "submission_time",
        "duration",
        "cpu_count",
        "cpu_capacity",
        "mem_capacity",
        "gpu_count",
        "gpu_capacity",
    ]
    assert fragments_table.column_names == [
        "id",
        "duration",
        "cpu_count",
        "cpu_usage",
        "gpu_count",
        "gpu_usage",
    ]
    assert tasks_table.num_rows == 1
    assert fragments_table.num_rows > 1
    assert fragments_table.column("id").to_pylist().count(7) == fragments_table.num_rows
