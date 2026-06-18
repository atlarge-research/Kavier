"""Tests for the friendly GPU/LLM spec lookups (``library.lookup``).

These helpers wrap the raw ``GPU_SPEC_LIBRARY`` / ``LLM_SPEC_LIBRARY`` dicts so
that an unknown key produces an actionable error (naming the key and listing
valid choices) instead of a bare ``KeyError``. The raised ``UnknownSpecError``
must still subclass ``KeyError`` so existing ``except KeyError`` handling works.
"""

from __future__ import annotations

import pytest

from kavier_library.gpu import GPU_SPEC_LIBRARY
from kavier_library.llm import LLM_SPEC_LIBRARY
from kavier_library.lookup import UnknownSpecError, get_gpu, get_llm


def test_get_gpu_known_returns_same_object():
    name = next(iter(GPU_SPEC_LIBRARY))
    assert get_gpu(name) is GPU_SPEC_LIBRARY[name]


def test_get_llm_known_returns_same_object():
    name = next(iter(LLM_SPEC_LIBRARY))
    assert get_llm(name) is LLM_SPEC_LIBRARY[name]


def test_get_gpu_unknown_raises_friendly_error():
    with pytest.raises(UnknownSpecError) as excinfo:
        get_gpu("NOPE-NO-SUCH-GPU")
    msg = str(excinfo.value)
    assert "NOPE-NO-SUCH-GPU" in msg
    assert "Available GPUs" in msg
    # Lists at least one real available key.
    assert next(iter(GPU_SPEC_LIBRARY)) in msg


def test_get_llm_unknown_raises_friendly_error():
    with pytest.raises(UnknownSpecError) as excinfo:
        get_llm("no-such-model")
    msg = str(excinfo.value)
    assert "no-such-model" in msg
    assert "Available models" in msg
    assert next(iter(LLM_SPEC_LIBRARY)) in msg


def test_unknown_spec_error_is_keyerror_subclass():
    # Backward-compat: callers catching KeyError must still catch this.
    assert issubclass(UnknownSpecError, KeyError)
    with pytest.raises(KeyError):
        get_gpu("still-not-a-gpu")


def test_engine_unknown_model_raises_friendly_error():
    from kavier_training.core.engine import simulate_training_step

    with pytest.raises(UnknownSpecError) as excinfo:
        simulate_training_step(
            model_name="not-a-real-model",
            gpu_model=next(iter(GPU_SPEC_LIBRARY)),
            tokens_per_sample=128,
            batch_size=1,
            method="full",
        )
    assert "not-a-real-model" in str(excinfo.value)


def test_engine_unknown_gpu_raises_friendly_error():
    from kavier_training.core.engine import simulate_training_step

    with pytest.raises(UnknownSpecError) as excinfo:
        simulate_training_step(
            model_name=next(iter(LLM_SPEC_LIBRARY)),
            gpu_model="not-a-real-gpu",
            tokens_per_sample=128,
            batch_size=1,
            method="full",
        )
    assert "not-a-real-gpu" in str(excinfo.value)
