"""Guardrail: locks the public surface Coastline imports from Kavier.

Coastline links to Kavier via PYTHONPATH=kavier/src (no pip install of kavier),
importing the specific symbols below and reaching into calibration._CAL to swap
calibration tables in and out for recalibration ablations. If any of these
import paths or the _CAL-swap contract change, Coastline breaks at import/run
time. This test fails loudly inside Kavier's own CI before that can happen.

References (in the sibling coastline/ tree):
    coastline/recommender/cli_interactive.py  -- PYTHONPATH wiring
    coastline/benchmark/kavier_calibration.py -- cal._CAL swap
"""

import pytest


def test_engine_symbols_importable():
    from kavier_training.core.engine import simulate_full_training, simulate_training_step

    assert callable(simulate_training_step)
    assert callable(simulate_full_training)


def test_io_library_and_opendc_symbols_importable():
    from kavier_io.training_opendc import build_training_opendc_frames
    from kavier_library.gpu import GPU_SPEC_LIBRARY
    from kavier_library.llm import LLM_SPEC_LIBRARY
    from kavier_opendc.adapter import prepare_opendc_input

    assert callable(build_training_opendc_frames)
    assert callable(prepare_opendc_input)
    assert GPU_SPEC_LIBRARY, "GPU_SPEC_LIBRARY must be non-empty"
    assert LLM_SPEC_LIBRARY, "LLM_SPEC_LIBRARY must be non-empty"


def test_calibration_cal_swap_contract():
    """Coastline does `cal._CAL = experimental_dict` then restores it. The getters
    must dereference the module global live (not cache values at import)."""
    import kavier_training.core.calibration as cal

    # Materialise the lazily-loaded tables before snapshotting (the loader is
    # lazy: `_CAL` is None until the first accessor call).
    cal.get_comm_scale()
    saved = cal._CAL
    try:
        cal._CAL = {**saved, "comm_scale": 0.123456}
        assert cal.get_comm_scale() == pytest.approx(0.123456)
    finally:
        cal._CAL = saved
    assert cal.get_comm_scale() == pytest.approx(float(saved["comm_scale"]))


def test_calibration_data_file_present():
    import kavier_training.core.calibration as cal

    # Located via the importlib.resources API (no `__file__`-relative path).
    resource = cal.files(cal._CALIBRATION_PACKAGE).joinpath(*cal._CALIBRATION_RESOURCE)
    assert resource.is_file()
    assert resource.name == "calibration.json"
