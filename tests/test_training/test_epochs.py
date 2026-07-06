"""Job sizing by epochs + dataset size (derives total_tokens)."""

import pytest

from kavier.sdk.training.core.engine import _resolve_total_tokens, simulate_full_training

_CFG = dict(
    model_name="mistral-7b-v0.1",
    method="lora",
    gpu_model="NVIDIA-A100-SXM4-80GB",
    tokens_per_sample=1024,
    batch_size=4,
    number_gpus=8,
    number_nodes=1,
)


# --- _resolve_total_tokens: the epochs/dataset -> total_tokens resolver ---


def test_explicit_total_tokens_beats_epochs_times_dataset():
    # total_tokens=5000 is returned even though epochs*dataset=3*1000=3000 differs.
    # 5000 != 3000 so a "return the product" mutation is caught.
    assert _resolve_total_tokens(5_000, 3, 1_000) == 5_000


def test_epochs_times_dataset_is_a_product():
    # 2 * 1_000_000 = 2_000_000 (product, not sum which would be 1_000_002).
    assert _resolve_total_tokens(None, 2, 1_000_000) == 2_000_000


def test_fractional_product_is_rounded_not_truncated():
    # 0.15 * 10 = 1.5 -> round() -> 2; plain int() truncation would give 1.
    assert _resolve_total_tokens(None, 0.15, 10) == 2


def test_none_when_neither_epochs_nor_dataset_given():
    assert _resolve_total_tokens(None, None, None) is None


@pytest.mark.parametrize(
    "epochs,dataset",
    [
        (2, None),  # dataset missing
        (None, 1_000),  # epochs missing
    ],
)
def test_epochs_and_dataset_must_be_supplied_together(epochs, dataset):
    # Exactly one of the pair is under-specified: cannot form a product.
    with pytest.raises(ValueError):
        _resolve_total_tokens(None, epochs, dataset)


@pytest.mark.parametrize(
    "epochs,dataset",
    [
        (-1, 1_000),  # negative epochs
        (2, -1),  # negative dataset
    ],
)
def test_negative_epochs_or_dataset_rejected(epochs, dataset):
    with pytest.raises(ValueError):
        _resolve_total_tokens(None, epochs, dataset)


# --- simulate_full_training: epochs plumbing into runtime ---


def test_epochs_over_dataset_equals_equivalent_total_tokens():
    # 2 epochs over 5M tokens describes the SAME 10M-token job as total_tokens=10M,
    # so both entry points must yield an identical runtime.
    by_epochs = simulate_full_training(**_CFG, epochs=2, dataset_tokens=5_000_000)
    by_tokens = simulate_full_training(**_CFG, total_tokens=10_000_000)
    assert by_epochs["total_tokens"] == 10_000_000  # product actually resolved
    assert by_epochs["train_runtime"] == pytest.approx(by_tokens["train_runtime"])
    assert by_epochs["train_runtime"] > 0


def test_runtime_is_total_tokens_divided_by_throughput():
    # Cross-check the two independently-returned fields: runtime = total_tokens / tps.
    # Catches a wrong divisor (e.g. an accidental *1000 / seconds<->ms mixup).
    out = simulate_full_training(**_CFG, epochs=2, dataset_tokens=5_000_000)
    expected = out["total_tokens"] / out["train_tokens_per_second"]
    assert out["train_runtime"] == pytest.approx(expected)


def test_runtime_scales_linearly_with_epochs():
    # tps is independent of job size, so 2x the epochs => 2x the runtime.
    one = simulate_full_training(**_CFG, epochs=1, dataset_tokens=5_000_000)
    two = simulate_full_training(**_CFG, epochs=2, dataset_tokens=5_000_000)
    assert two["train_runtime"] == pytest.approx(2 * one["train_runtime"])


def test_no_job_size_yields_zero_runtime():
    # Neither total_tokens nor epochs/dataset -> total_tokens None -> runtime 0.0 (not an error).
    out = simulate_full_training(**_CFG)
    assert out["total_tokens"] is None
    assert out["train_runtime"] == 0.0
