# `kavier.sdk.training.calibration`

The fitted calibration table(s) + the engine that regenerates them. Two layers:

- **Tier 1 (global scales)** = Powell minimization (derivative-free, not gradient) in log-space,
  minimizing median-APE + an L2 pull toward neutral, λ chosen on validation.
- **Tier 2 (per-config)** = median of measured/predicted ratios.

## Files

- `__init__.py` — the **lean runtime accessor** (`get_*`, `use_calibration`, `available_calibrations`).
  Stdlib-only: importing it never pulls in scipy/sklearn/numpy/pandas.
- `calibration.json` — the shipped **default** table (== the 6-model fit).
- `versions/` — the selectable from-scratch fits:
  - `calibration_4model.json` — dense-4 only (`mistral-7b-v0.1`, `granite-3.3-8b`, `granite-3-8b`,
    `llama3.2-3b`); the exp1 head-to-head set.
  - `calibration_6model.json` — dense-4 + `granite-3.1-2b` + `granite-3.1-8b-instruct`; byte-identical
    to `calibration.json`.
- `engine.py` — **the one** dev-only fitter file: from-scratch `regenerate`, the Powell Tier-1 fit,
  and the multi-GPU-correction fit. Needs the `[calibration]` extra (scipy/sklearn).

## Selecting a calibration at runtime

```python
import kavier.sdk.training.calibration as cal
cal.available_calibrations()      # ['default', '4model', '6model']
cal.use_calibration("4model")     # or "6model" / "default" / a path to a .json file
```

Or set `KAVIER_CALIBRATION=4model` (a name or path) before first use. Default = `calibration.json`.

## Regenerating (needs the profiling trace — not vendored)

```bash
python -m kavier.sdk.training.calibration.engine --check               # rebuild BOTH sets; verify byte-identical
python -m kavier.sdk.training.calibration.engine --model-set 4 --write  # rewrite versions/calibration_4model.json
python -m kavier.sdk.training.calibration.engine --write                # rebuild + overwrite the shipped 6-model default
```

⚠ `multi_gpu_correction` >8 GPU is extrapolation → the recommender stays ≤8 GPU.
