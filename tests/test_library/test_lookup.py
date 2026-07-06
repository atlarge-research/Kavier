"""Friendly GPU/LLM spec lookups (library.lookup): unknown keys raise an actionable error naming
the key and valid choices. UnknownSpecError must subclass KeyError so existing handlers still catch it."""

from __future__ import annotations

import pytest

from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY
from kavier.sdk.library.llm import LLM_SPEC_LIBRARY
from kavier.sdk.library.lookup import UnknownSpecError, get_gpu, get_llm


# --- known keys: the lookup is a faithful, identity-preserving view of the library ---
# Oracle: the library dict IS the source of truth; a correct lookup must return that exact
# object for every catalog key. Parametrized over the whole catalog so a lookup that returned
# a fixed/wrong entry (e.g. always the first spec) goes red on every other key.
@pytest.mark.parametrize("name", sorted(GPU_SPEC_LIBRARY))
def test_get_gpu_returns_the_library_object_for_every_key(name):
    assert get_gpu(name) is GPU_SPEC_LIBRARY[name]


@pytest.mark.parametrize("name", sorted(LLM_SPEC_LIBRARY))
def test_get_llm_returns_the_library_object_for_every_key(name):
    assert get_llm(name) is LLM_SPEC_LIBRARY[name]


# --- unknown keys: actionable error naming the key, the kind, the count, and ALL choices ---
def test_get_gpu_unknown_error_names_key_kind_count_and_all_choices():
    with pytest.raises(UnknownSpecError) as excinfo:
        get_gpu("NOPE-NO-SUCH-GPU")
    msg = str(excinfo.value)
    assert "NOPE-NO-SUCH-GPU" in msg  # the offending key
    assert "Available GPUs" in msg  # names the kind (GPU, not model)
    # Count is the library size (independent oracle = the catalog itself), not a hard-coded int.
    assert f"({len(GPU_SPEC_LIBRARY)})" in msg
    # The whole valid set is listed, so a truncated/partial message goes red.
    for gpu_name in GPU_SPEC_LIBRARY:
        assert gpu_name in msg


def test_get_llm_unknown_error_names_key_kind_count_and_all_choices():
    with pytest.raises(UnknownSpecError) as excinfo:
        get_llm("no-such-model")
    msg = str(excinfo.value)
    assert "no-such-model" in msg
    assert "Available models" in msg  # "model" kind, not "GPU"
    assert f"({len(LLM_SPEC_LIBRARY)})" in msg
    for llm_name in LLM_SPEC_LIBRARY:
        assert llm_name in msg


# --- backward-compat: callers catching KeyError must still catch this ---
def test_unknown_spec_error_is_keyerror_subclass():
    assert issubclass(UnknownSpecError, KeyError)
    with pytest.raises(KeyError):
        get_gpu("still-not-a-gpu")


def test_str_does_not_requote_message_unlike_plain_keyerror():
    # KeyError.__str__ wraps its arg in repr() (adds surrounding quotes); the custom __str__
    # override must return the raw message so the actionable text reads cleanly.
    err = UnknownSpecError("GPU", "X", ["a", "b"])
    raw = err.args[0]
    assert str(err) == raw
    assert not str(err).startswith('"')
    # Contrast proves the override is load-bearing: a plain KeyError re-quotes the same text.
    assert str(KeyError(raw)) != raw


# --- integration: the training engine surfaces the same friendly error, not a bare KeyError ---
# get_llm is called before get_gpu in simulate_training_step, so each test isolates one lookup
# by making only that argument invalid.
def test_engine_unknown_model_raises_friendly_error():
    from kavier.sdk.training.core.engine import simulate_training_step

    with pytest.raises(UnknownSpecError) as excinfo:
        simulate_training_step(
            model_name="not-a-real-model",
            gpu_model=next(iter(GPU_SPEC_LIBRARY)),  # valid GPU: failure must be the model
            tokens_per_sample=128,
            batch_size=1,
            method="full",
        )
    msg = str(excinfo.value)
    assert "not-a-real-model" in msg
    assert "Available models" in msg


def test_engine_unknown_gpu_raises_friendly_error():
    from kavier.sdk.training.core.engine import simulate_training_step

    with pytest.raises(UnknownSpecError) as excinfo:
        simulate_training_step(
            model_name=next(iter(LLM_SPEC_LIBRARY)),  # valid model: failure must be the GPU
            gpu_model="not-a-real-gpu",
            tokens_per_sample=128,
            batch_size=1,
            method="full",
        )
    msg = str(excinfo.value)
    assert "not-a-real-gpu" in msg
    assert "Available GPUs" in msg
