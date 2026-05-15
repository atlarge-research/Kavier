from __future__ import annotations

import pytest

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
