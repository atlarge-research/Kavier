# tools/ — developer-only utilities (not shipped)

Everything under `tools/` is **development-only**. It is never packaged, installed, or
imported by the `kavier` distribution — moving or deleting it cannot change the
installed package's behaviour. It exists for provenance and reproducibility: showing,
end to end, how Kavier's shipped numbers were produced.

## `tools/calibration/` — calibration fitters

These scripts (re)fit the **empirical (Tier-2) calibration** terms in the shipped
`src/kavier_training/data/calibration.json` from measured profiling throughput, and
document how each term was derived. The physics-level terms (`comm_scale`,
`mfu_multiplier`) are fit elsewhere (the Powell joint fitter in
`coastline/benchmark/kavier_calibration.py`) and are left untouched here — leaving
`mfu_multiplier` alone keeps Kavier's *power/energy* prediction bit-identical.

| script | fits | term(s) written |
| --- | --- | --- |
| `fit_calibration.py` | full Tier-2 residual on the curated 6-model 85% train+val split, evaluated on the held-out 15% test | a fresh `calibration_6models_85.json` (mgc / `method_scale` / `model_scale` / `interaction_scale`) |
| `fit_invitro_calibration.py` | per-model `model_scale` + `interaction_scale` for specific models (e.g. the rwt3-llmbuild in-vitro granite models) | those models' entries in `calibration.json` |
| `fit_multi_gpu_correction.py` | the multi-node (`>8` GPU) `multi_gpu_correction.by_num_gpus` anchors | those GPU-count entries in `calibration.json` |

All three share `_common.py` (path bootstrap, `CAL_PATH`, `mdape`, the
`calibration_override` context manager, and the common `--trace`/`--write` argparse base).
Every fitter uses the same median-ratio convention the engine applies, and runs as a
**dry-run by default** — it prints the fit and held-out MdAPE but writes nothing unless
you pass `--write`.

### Data is NOT vendored

The fitters need IBM-internal **ado-sfttrainer / PD1 profiling traces**, which are *not*
committed to this repo (size + dataset licence). They are expected as siblings of the
kavier checkout, under `../trace-archive/pd1-profiling-dataset/` — pass an explicit path
with `--trace` if yours live elsewhere. Required CSV columns: `model_name`, `gpu_model`,
`method`, `number_gpus`, `number_nodes`, `batch_size`, `tokens_per_sample`, `is_valid`,
`dataset_tokens_per_second`. `fit_calibration.py` additionally needs the `coastline`
checkout (`../coastline`, for `trainer.common`) and its `scikit-learn` dependency, so the
70/15/15 split byte-matches the ML trainers'.

Without the traces the fitters cannot run, but they still import cleanly
(`python tools/calibration/fit_invitro_calibration.py --help`).

### Example commands

```bash
# Refit the full Tier-2 residual on the 6-model 85% split; print in-sample + held-out MdAPE.
python tools/calibration/fit_calibration.py
python tools/calibration/fit_calibration.py --write   # also write calibration_6models_85.json

# Add/refresh per-model calibration for specific models, then write into calibration.json.
python tools/calibration/fit_invitro_calibration.py \
    --trace ../trace-archive/pd1-profiling-dataset/ado-sfttrainer-for_invitro.csv \
    --models granite-3.1-2b granite-3.1-8b-instruct --write

# Refit the multi-node (>8 GPU) correction anchors (dry-run, then write).
python tools/calibration/fit_multi_gpu_correction.py \
    --trace ../trace-archive/pd1-profiling-dataset/ado-sfttrainer-raw.csv
python tools/calibration/fit_multi_gpu_correction.py \
    --trace ../trace-archive/pd1-profiling-dataset/ado-sfttrainer-raw.csv --counts 16 32 128 --write
```

Omit `--write` for a dry run (print only; touches nothing).
