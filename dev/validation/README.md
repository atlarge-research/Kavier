# Validation harness (experimental, dev-only)

Experimental harness that checks `kavier.sdk.training.core.engine` predictions against measured throughput
(`validator.py`, `run_benchmarks.py`). It requires `validation_clean.csv` — the internal, unshipped
held-out measurement set behind the published ~6.2% MdAPE figure (see `docs/content/known-weaknesses.md`) —
which the scripts resolve to `dev/data/input/validation_clean.csv`; run from the repo root with `PYTHONPATH=src`.
Not part of the published `kavier` package and not covered by CI's strict-mypy gate.
