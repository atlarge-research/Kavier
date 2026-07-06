"""Kavier SDK: the simulation engines behind the public ``kavier.inference`` / ``kavier.training`` verbs.

Submodules: ``inference`` (per-request inference sim), ``training`` (analytical training model +
calibration), ``energy`` (GPU power/efficiency), ``co2`` (carbon), ``io`` (trace/OpenDC I/O), and
``library`` (static GPU/LLM specs).

Import-light by contract: keep this module (and ``kavier.sdk.training``) free of eager engine/facade
imports. The calibration accessor ``kavier.sdk.training.calibration`` MUST stay importable without
pulling scipy/sklearn/numpy/pandas (guarded by
``test_calibration_versions.py::test_accessor_import_is_stdlib_only``); importing it executes this
package's ``__init__``, so any heavy import added here would silently break that contract.
"""
