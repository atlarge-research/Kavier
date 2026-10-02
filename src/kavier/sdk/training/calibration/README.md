# `kavier.sdk.training.calibration`

Fitted calibration tables and the engine that regenerates them. The fit has two tiers:

- Tier 1 (global scales): Powell minimization in log-space (derivative-free) of median APE plus an L2
  pull toward neutral, with $\lambda$ chosen on the validation split.
- Tier 2 (per-config): median of measured/predicted ratios.

## Files

- `__init__.py`: runtime accessor (`get_*`, `use_calibration`, `available_calibrations`). It is
  stdlib-only; importing it does not load scipy, sklearn, numpy or pandas.
- `calibration.json`: the shipped default table, identical to the 6-model fit.
- `versions/`: the selectable fits.
  - `calibration_4model.json`: dense-4 only (`mistral-7b-v0.1`, `granite-3.3-8b`, `granite-3-8b`,
    `llama3.2-3b`), the Exp1 head-to-head set.
  - `calibration_6model.json`: dense-4 plus `granite-3.1-2b` and `granite-3.1-8b-instruct`;
    byte-identical to `calibration.json`.
- `engine.py`: the dev-only fitter, with from-scratch `regenerate`, the Tier 1 Powell fit and the
  multi-GPU correction fit. Needs the `[calibration]` extra (scipy, sklearn).

## Selecting a calibration at runtime

```python
import kavier.sdk.training.calibration as cal
cal.available_calibrations()      # ['default', '4model', '6model']
cal.use_calibration("4model")     # or "6model" / "default" / a path to a .json file
```

Or set `KAVIER_CALIBRATION=4model` (a name or path) before first use. The default is `calibration.json`.

## Regenerating

Regeneration needs the profiling trace, which is not vendored.

```bash
python -m kavier.sdk.training.calibration.engine --check               # rebuild both sets; verify byte-identical
python -m kavier.sdk.training.calibration.engine --model-set 4 --write  # rewrite versions/calibration_4model.json
python -m kavier.sdk.training.calibration.engine --write                # rebuild + overwrite the shipped 6-model default
```

`multi_gpu_correction` above 8 GPUs is an extrapolation, so the recommender stays at 8 GPUs or fewer.
