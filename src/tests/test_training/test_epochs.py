"""Job sizing by epochs + dataset size (derives total_tokens)."""

import pytest

from kavier_training.core.engine import _resolve_total_tokens, simulate_full_training

_CFG = dict(
    model_name="mistral-7b-v0.1",
    method="lora",
    gpu_model="NVIDIA-A100-SXM4-80GB",
    tokens_per_sample=1024,
    batch_size=4,
    number_gpus=8,
    number_nodes=1,
)


def test_resolve_prefers_explicit_total_tokens():
    # total_tokens wins even if epochs/dataset are also supplied.
    assert _resolve_total_tokens(5_000, 3, 1_000) == 5_000


def test_resolve_epochs_times_dataset():
    assert _resolve_total_tokens(None, 2, 1_000_000) == 2_000_000


def test_resolve_none_when_unset():
    assert _resolve_total_tokens(None, None, None) is None


def test_resolve_requires_epochs_and_dataset_together():
    with pytest.raises(ValueError):
        _resolve_total_tokens(None, 2, None)
    with pytest.raises(ValueError):
        _resolve_total_tokens(None, None, 1_000)


def test_resolve_rejects_negative():
    with pytest.raises(ValueError):
        _resolve_total_tokens(None, -1, 1_000)


def test_epochs_match_equivalent_total_tokens():
    # 2 epochs over 5M tokens == 10M tokens: identical runtime.
    by_epochs = simulate_full_training(**_CFG, epochs=2, dataset_tokens=5_000_000)
    by_tokens = simulate_full_training(**_CFG, total_tokens=10_000_000)
    assert by_epochs["train_runtime"] == pytest.approx(by_tokens["train_runtime"])
    assert by_epochs["train_runtime"] > 0
