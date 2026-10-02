"""The public ``kavier`` namespace.

``import kavier`` lazily re-exports the predictor verbs and the top-level spec and engine symbols
(PEP 562 ``__getattr__`` in ``kavier/__init__.py``); the engines live under ``kavier.sdk.*``. The tests
check identity: a lazy re-export returns the same object as its sdk source, and the calibration accessor
is one module object under both spellings, so a live ``calibration._CAL`` swap is visible through either.
"""

from __future__ import annotations

import importlib
import pathlib
import tomllib

import pytest


def test_top_level_symbols_are_the_same_objects_as_their_sdk_sources():
    # A re-exported copy, a wrapper, or a name bound to the wrong module fails the ``is`` checks.
    from kavier import (
        GPU_SPEC_LIBRARY,
        LLM_SPEC_LIBRARY,
        simulate_full_training,
        simulate_training_step,
    )
    from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY as sdk_gpu
    from kavier.sdk.library.llm import LLM_SPEC_LIBRARY as sdk_llm
    from kavier.sdk.training.core.engine import (
        simulate_full_training as sdk_full,
    )
    from kavier.sdk.training.core.engine import (
        simulate_training_step as sdk_step,
    )

    assert simulate_training_step is sdk_step
    assert simulate_full_training is sdk_full
    assert GPU_SPEC_LIBRARY is sdk_gpu
    assert LLM_SPEC_LIBRARY is sdk_llm


def test_inference_and_training_aliases_are_the_sdk_verb_packages():
    # ``kavier.inference`` and ``kavier.training`` are the sdk package objects, and the four batch
    # predictors resolve through the package's lazy ``__getattr__``. A wrong ``_LAZY_ALIASES`` entry
    # breaks ``is``; a verb missing from ``_FACADE_EXPORTS`` makes ``getattr`` raise AttributeError.
    import kavier
    import kavier.sdk.inference
    import kavier.sdk.training

    assert kavier.inference is kavier.sdk.inference
    assert kavier.training is kavier.sdk.training
    for verb in ("performance", "energy", "efficiency", "carbon"):
        assert callable(getattr(kavier.inference, verb))
        assert callable(getattr(kavier.training, verb))


def test_documented_sdk_engine_packages_are_importable():
    # Every engine package named in the ``kavier.sdk`` docstring imports under that name. A removed or
    # renamed package raises ImportError; a wrong alias fails the ``__name__`` check.
    for name in (
        "kavier.sdk.inference",
        "kavier.sdk.training",
        "kavier.sdk.energy",
        "kavier.sdk.co2",
        "kavier.sdk.io",
        "kavier.sdk.io.opendc",
        "kavier.sdk.library",
    ):
        module = importlib.import_module(name)
        assert module.__name__ == name


def test_version_matches_pyproject_single_source_of_truth():
    # Compares with pyproject's static ``version``, which ``kavier/__init__.py`` names as the version
    # source. Fails if pyproject is bumped without reinstalling, or if ``__version__`` is hardcoded wrong.
    import kavier

    if kavier.__version__.endswith("+unknown"):
        pytest.skip("source tree without installed dist metadata (clean-checkout fallback)")

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    assert kavier.__version__ == pyproject["project"]["version"]


def test_unknown_top_level_attribute_raises_attributeerror():
    # The PEP 562 ``__getattr__`` raises AttributeError for an unknown name.
    import kavier

    with pytest.raises(AttributeError):
        kavier.definitely_not_a_real_attribute  # noqa: B018


def test_calibration_accessor_is_one_object_across_both_spellings():
    # ``kavier.sdk.training.calibration`` and the aliased ``kavier.training.calibration`` are one module
    # object, so a live ``_CAL`` swap is seen through either spelling.
    import kavier
    import kavier.sdk.training.calibration as direct

    aliased = kavier.training.calibration
    assert aliased is direct

    # Load the lazy table, swap it, and read the change back through the other spelling.
    direct.get_comm_scale()
    saved = direct._CAL
    try:
        direct._CAL = {**saved, "comm_scale": 0.123456}
        assert aliased.get_comm_scale() == pytest.approx(0.123456)
    finally:
        direct._CAL = saved
    assert aliased.get_comm_scale() == pytest.approx(float(saved["comm_scale"]))
