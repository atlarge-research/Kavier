"""Guardrail for the public ``kavier.sdk.*`` surface: the engine / IO / library dotted import paths
external consumers bind to (module + symbol + the load-bearing keyword parameters), the spec-library
exact-match keying contract, and the live ``calibration._CAL`` swap (getters must dereference the
module global on every call, so a caller can swap the table in and out).

Behavioural depth (physics magnitudes, calibration fallbacks, opendc round-trips) lives in the
dedicated unit suites; this file only locks the *shape* of the public API so a rename/move is caught."""

from __future__ import annotations

import importlib
import inspect

import pytest

# (dotted module, public callable, the keyword params consumers bind by name). Renaming/moving any of
# these — or renaming a documented kwarg — is a breaking change to the public API and must go red here.
_PUBLIC_CALLABLES = [
    (
        "kavier.sdk.training.core.engine",
        "simulate_training_step",
        ("model_name", "gpu_model", "method", "batch_size", "tokens_per_sample"),
    ),
    (
        "kavier.sdk.training.core.engine",
        "simulate_full_training",
        ("model_name", "gpu_model", "method", "number_gpus", "number_nodes"),
    ),
    (
        "kavier.sdk.io.opendc.adapter",
        "prepare_opendc_input",
        ("tasks", "fragments", "dst_dir"),
    ),
]


@pytest.mark.parametrize(
    ("module_path", "symbol", "required_kwargs"),
    _PUBLIC_CALLABLES,
    ids=[f"{m.rsplit('.', 1)[-1]}.{s}" for m, s, _ in _PUBLIC_CALLABLES],
)
def test_public_callable_surface(module_path: str, symbol: str, required_kwargs: tuple[str, ...]) -> None:
    # Falsifier: move the module, rename the symbol, or rename any listed kwarg -> this param goes red.
    module = importlib.import_module(module_path)
    fn = getattr(module, symbol)
    assert callable(fn), f"{module_path}.{symbol} must be callable"
    params = inspect.signature(fn).parameters
    missing = [name for name in required_kwargs if name not in params]
    assert not missing, f"{module_path}.{symbol} lost public kwargs: {missing}"


@pytest.mark.parametrize(
    ("module_path", "symbol", "name_attr"),
    [
        ("kavier.sdk.library.gpu", "GPU_SPEC_LIBRARY", "name"),
        ("kavier.sdk.library.llm", "LLM_SPEC_LIBRARY", "name"),
    ],
    ids=["gpu", "llm"],
)
def test_spec_library_surface_is_nonempty_and_self_keyed(module_path: str, symbol: str, name_attr: str) -> None:
    # The libraries are the public catalogue and lookups are exact-match on the dict key, so each key
    # MUST equal its own spec's name. Oracle = that invariant (independent of any snapshot).
    # Falsifier: an empty library, or a mislabelled entry ({"A100-80GB": spec(name="A100")}) -> red.
    library = getattr(importlib.import_module(module_path), symbol)
    assert library, f"{symbol} must be non-empty"
    mismatched = {key: getattr(spec, name_attr) for key, spec in library.items() if key != getattr(spec, name_attr)}
    assert not mismatched, f"{symbol} keys must equal spec.{name_attr}; mismatched: {mismatched}"


def test_calibration_getters_dereference_cal_live() -> None:
    # Headline contract: the getters read the module global on EVERY call, so a consumer can swap the
    # table in (and restore it) around a block. Oracle = the sentinel 0.123456 we install by hand.
    import kavier.sdk.training.calibration as cal

    # _CAL is None until first access; materialise it so we snapshot the real loaded table.
    cal.get_comm_scale()
    saved = cal._CAL
    assert saved is not None
    original = float(saved["comm_scale"])
    try:
        cal._CAL = {**saved, "comm_scale": 0.123456}
        # Falsifier: if get_comm_scale cached the value instead of re-reading _CAL, this stays `original`.
        assert cal.get_comm_scale() == pytest.approx(0.123456)
    finally:
        cal._CAL = saved
    # Falsifier: a getter that mutated/failed to restore the shared table would break this restore check.
    assert cal.get_comm_scale() == pytest.approx(original)
