"""Simulation engines behind ``kavier.inference`` and ``kavier.training``.

Submodules: ``inference`` (per-request inference simulator), ``training`` (analytical training
model and calibration), ``energy`` (GPU power and efficiency), ``co2`` (carbon), ``io`` (trace and
OpenDC I/O), and ``library`` (static GPU and LLM specs).

Keep this module and ``kavier.sdk.training`` free of engine imports. Importing
``kavier.sdk.training.calibration`` runs this file, and that import must not load scipy, sklearn,
numpy or pandas (tested by ``test_calibration_versions.py::test_accessor_import_is_stdlib_only``).
"""
