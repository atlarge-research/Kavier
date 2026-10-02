"""GPU/LLM spec lookups (library.lookup): an unknown key raises an error naming the key and the valid
choices. UnknownSpecError subclasses KeyError, so existing handlers still catch it."""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor

import pytest

from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY
from kavier.sdk.library.llm import LLM_SPEC_LIBRARY
from kavier.sdk.library.lookup import UnknownSpecError, get_gpu, get_llm


# --- known keys: the lookup returns the library object itself ---
# Parametrized over the whole catalog, so a lookup that returns a fixed entry fails on the other keys.
@pytest.mark.parametrize("name", sorted(GPU_SPEC_LIBRARY))
def test_get_gpu_returns_the_library_object_for_every_key(name):
    assert get_gpu(name) is GPU_SPEC_LIBRARY[name]


@pytest.mark.parametrize("name", sorted(LLM_SPEC_LIBRARY))
def test_get_llm_returns_the_library_object_for_every_key(name):
    assert get_llm(name) is LLM_SPEC_LIBRARY[name]


# --- unknown keys: the error names the key, the kind, the count and every choice ---
def test_get_gpu_unknown_error_names_key_kind_count_and_all_choices():
    with pytest.raises(UnknownSpecError) as excinfo:
        get_gpu("NOPE-NO-SUCH-GPU")
    msg = str(excinfo.value)
    assert "NOPE-NO-SUCH-GPU" in msg  # the offending key
    assert "Available GPUs" in msg
    # count is the library size
    assert f"({len(GPU_SPEC_LIBRARY)})" in msg
    # every valid key is listed
    for gpu_name in GPU_SPEC_LIBRARY:
        assert gpu_name in msg


def test_get_llm_unknown_error_names_key_kind_count_and_all_choices():
    with pytest.raises(UnknownSpecError) as excinfo:
        get_llm("no-such-model")
    msg = str(excinfo.value)
    assert "no-such-model" in msg
    assert "Available models" in msg
    assert f"({len(LLM_SPEC_LIBRARY)})" in msg
    for llm_name in LLM_SPEC_LIBRARY:
        assert llm_name in msg


# --- callers catching KeyError still catch UnknownSpecError ---
def test_unknown_spec_error_is_keyerror_subclass():
    assert issubclass(UnknownSpecError, KeyError)
    with pytest.raises(KeyError):
        get_gpu("still-not-a-gpu")


def test_str_does_not_requote_message_unlike_plain_keyerror():
    # KeyError.__str__ wraps its argument in repr() and adds quotes; the __str__ override returns
    # the raw message.
    err = UnknownSpecError("GPU", "X", ["a", "b"])
    raw = err.args[0]
    assert str(err) == raw
    assert not str(err).startswith('"')
    # a plain KeyError quotes the same text
    assert str(KeyError(raw)) != raw


# --- integration: the training engine raises the same error ---
# simulate_training_step calls get_llm before get_gpu, so each test makes only one argument invalid.
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


# --- pickling: process-pool workers send exceptions back to the parent by pickle ---
def test_unknown_spec_error_survives_a_pickle_round_trip():
    err = UnknownSpecError("GPU", "X", ["a", "b"])
    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is UnknownSpecError
    assert str(restored) == str(err)
    assert restored.args == err.args


def test_unknown_gpu_in_a_process_pool_worker_reaches_the_caller():
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(get_gpu, "not-a-real-gpu")
        with pytest.raises(UnknownSpecError, match="not-a-real-gpu"):
            future.result(timeout=120)
