"""The public ``kavier`` namespace: ``import kavier`` lazily re-exports the predictor verbs and the
top-level spec/engine symbols (PEP-562 ``__getattr__`` in ``kavier/__init__.py``), and the engines live
under ``kavier.sdk.*``. The load-bearing contracts here are IDENTITY, not mere importability: a lazy
re-export must hand back the *same* object as its sdk source (never a copy/wrapper), and the calibration
accessor must remain ONE module object across its two spellings so the live ``calibration._CAL`` swap is
visible through either — a copy would silently split the table.
"""

from __future__ import annotations

import importlib
import pathlib
import tomllib

import pytest


def test_top_level_symbols_are_the_same_objects_as_their_sdk_sources():
    # Contract: the lazy re-export delegates to the sdk object; it must NOT clone or wrap it.
    # Falsification: if ``kavier/__init__.py`` re-exported a copy (e.g. wrapped the fn or rebuilt the
    # dict), or pointed a name at the wrong module, ``is`` goes False. Identity, not just callable.
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
    # ``kavier.inference`` / ``kavier.training`` are aliases (same module object), not fresh imports;
    # and the four batch predictors resolve via the package lazy ``__getattr__`` on both spellings.
    # Falsification: a typo in ``_LAZY_ALIASES`` pointing elsewhere breaks ``is``; a verb dropped from
    # a package's ``_FACADE_EXPORTS`` makes ``getattr`` raise AttributeError below.
    import kavier
    import kavier.sdk.inference
    import kavier.sdk.training

    assert kavier.inference is kavier.sdk.inference
    assert kavier.training is kavier.sdk.training
    for verb in ("performance", "energy", "efficiency", "carbon"):
        assert callable(getattr(kavier.inference, verb))
        assert callable(getattr(kavier.training, verb))


def test_documented_sdk_engine_packages_are_importable():
    # Surface lock: every engine package named in the ``kavier.sdk`` docstring must exist and import
    # to the module of that exact name. Falsification: rename/remove any (e.g. ``kavier.sdk.co2``)
    # -> ImportError; a mis-alias returning a differently-named module -> ``__name__`` mismatch.
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
    # Oracle independent of the runtime attribute: pyproject's static ``version`` (declared the single
    # source of truth in ``kavier/__init__.py``). Falsification: bump pyproject without rebuilding the
    # installed metadata, or hardcode a wrong ``__version__`` -> mismatch.
    import kavier

    if kavier.__version__.endswith("+unknown"):
        pytest.skip("source tree without installed dist metadata (clean-checkout fallback)")

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    assert kavier.__version__ == pyproject["project"]["version"]


def test_unknown_top_level_attribute_raises_attributeerror():
    # Error path of the PEP-562 ``__getattr__``: an unknown name is a hard AttributeError, not a
    # silent None or a swallowed miss. Falsification: a bare ``return None`` fallthrough -> no raise.
    import kavier

    with pytest.raises(AttributeError):
        kavier.definitely_not_a_real_attribute  # noqa: B018


def test_calibration_accessor_is_one_object_across_both_spellings():
    # The load-bearing contract: ``kavier.sdk.training.calibration`` and the aliased
    # ``kavier.training.calibration`` must be ONE module object, so a live ``_CAL`` swap is seen through
    # either. Falsification: if the alias produced a distinct module, the swap below would be invisible
    # via the other spelling and the ``is`` check would fail.
    import kavier
    import kavier.sdk.training.calibration as direct

    aliased = kavier.training.calibration
    assert aliased is direct

    # Materialise the lazily-loaded table, then swap it and confirm the change is visible through the
    # OTHER spelling — proving a single shared module global, not two copies.
    direct.get_comm_scale()
    saved = direct._CAL
    try:
        direct._CAL = {**saved, "comm_scale": 0.123456}
        assert aliased.get_comm_scale() == pytest.approx(0.123456)
    finally:
        direct._CAL = saved
    assert aliased.get_comm_scale() == pytest.approx(float(saved["comm_scale"]))
