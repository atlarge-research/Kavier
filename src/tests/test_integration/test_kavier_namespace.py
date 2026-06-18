"""Tests for the canonical ``kavier`` namespace package.

The ``kavier`` package is an additive re-export layer over the underlying
sibling packages (``kavier_training``, ``kavier_library``, ...). New consumers
should be able to ``import kavier`` and reach the public API, while every alias
must resolve to the *same* module/object as the underlying spelling (so there is
never a duplicate module that would, e.g., break the live ``calibration._CAL``
swap contract).
"""

from __future__ import annotations


def test_top_level_public_api_is_importable():
    import kavier
    from kavier import (
        GPU_SPEC_LIBRARY,
        LLM_SPEC_LIBRARY,
        simulate_full_training,
        simulate_training_step,
    )

    assert callable(simulate_training_step)
    assert callable(simulate_full_training)
    assert GPU_SPEC_LIBRARY
    assert LLM_SPEC_LIBRARY
    assert kavier.__version__


def test_top_level_symbols_are_identical_to_legacy():
    from kavier import (
        GPU_SPEC_LIBRARY,
        LLM_SPEC_LIBRARY,
        simulate_full_training,
        simulate_training_step,
    )
    from kavier_training.core.engine import (
        simulate_full_training as legacy_full,
    )
    from kavier_training.core.engine import (
        simulate_training_step as legacy_step,
    )
    from kavier_library.gpu import GPU_SPEC_LIBRARY as legacy_gpu
    from kavier_library.llm import LLM_SPEC_LIBRARY as legacy_llm

    assert simulate_training_step is legacy_step
    assert simulate_full_training is legacy_full
    assert GPU_SPEC_LIBRARY is legacy_gpu
    assert LLM_SPEC_LIBRARY is legacy_llm


def test_submodule_aliases_resolve_to_legacy_packages():
    import kavier

    expected = {
        "training": "kavier_training",
        "inference": "kavier_inference",
        "io": "kavier_io",
        "energy": "kavier_energy",
        "co2": "kavier_co2",
        "library": "kavier_library",
        "opendc": "kavier_opendc",
    }
    for alias, legacy_name in expected.items():
        module = getattr(kavier, alias)
        assert module.__name__ == legacy_name


def test_deep_imports_share_legacy_module_objects():
    # The function object reached through the kavier.* alias must be the exact
    # same object as via the legacy path (no duplicate module).
    from kavier.training.core.engine import simulate_training_step as via_alias

    from kavier_training.core.engine import simulate_training_step as via_legacy

    assert via_alias is via_legacy

    import kavier.library.gpu as alias_gpu

    from kavier_library.gpu import GPU_SPEC_LIBRARY as legacy_gpu

    assert alias_gpu.GPU_SPEC_LIBRARY is legacy_gpu


def test_calibration_module_identity_across_spellings():
    # Critical for the _CAL swap contract: both spellings must be one module.
    import kavier.training.core.calibration as via_alias

    import kavier_training.core.calibration as via_legacy

    assert via_alias is via_legacy
