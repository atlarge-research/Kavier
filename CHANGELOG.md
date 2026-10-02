# Changelog

All notable changes to Kavier are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Kavier adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The version lives in `pyproject.toml`;
release tags mirror it as `vX.Y.Z`.

## [Unreleased]

### Added

- The Zenodo DOI (10.5281/zenodo.23110996, all versions) in the README, `CITATION.cff` and the docs.

## [0.5.3] - 2026-10-02

Training and calibration predictions for the four calibrated GPUs are unchanged from 0.5.2.

### Added

- Documentation site (`mkdocs.yml`, `docs/`): the fine-tuning model, its calibration, usage, and the cluster
  simulator. CI builds it on every pull request; `docs-deploy.yml` publishes it to GitHub Pages.
- Citation metadata for the software in `CITATION.cff` and `.zenodo.json`.
- `qlora` and `alora` training methods, with LoRA trainable parameters and a neutral method scale.

### Changed

- Unknown training methods raise `ValueError` instead of running as full fine-tuning.
- `--config` values pass the same type and choice checks as flags, can satisfy required flags, and lose
  to flags given on the command line.
- Consolidated cluster policies with `--oversized drop` skip only jobs larger than the cluster; other
  jobs that do not fit their requested nodes are widened, as with `cap`.
- `kavier carbon --powersource` reads OpenDC's int64 millisecond timestamps, prefers
  `timestamp_absolute`, and bills each row over the interval that ends at its timestamp.
- OpenDC tasks from `kavier inference` start at 2026-01-01 00:00 UTC, so identical runs give identical
  workloads; a second run in the same second gets its own output folder.
- `tokens_per_step` is the full data-parallel product and `train_steps_per_second` includes the
  multi-GPU correction; throughput is unchanged.
- GPU catalog: dense FP16 peaks for L4 (121 TFLOPS), H100-SXM and H200 SXM (989), and H100-PCIe (756), and
  NVLink rates for H100-SXM and H200 SXM. Inference latency, energy, cost and carbon predictions change for
  these GPUs. The H100-PCIe MFU factor doubles, so its single-GPU training throughput is unchanged while its
  reported compute utilization and power rise. The calibrated entries are unchanged.
- `python -m kavier.sdk.training.calibration.engine --check` compares the values the model uses and
  ignores the >8-GPU multi-GPU corrections, which move with the catalog.
- CI runs on Linux only, on pull requests, with a time cap on every job and `uv sync --locked`;
  pre-commit and CI use the ruff pinned in the dev group.
- Releases check the version in `pyproject.toml`, `CITATION.cff` and `.zenodo.json`, and the GitHub
  release no longer waits for the PyPI upload.
- mypy `python_version` is 3.12, since the numpy 2.5 stubs use 3.12-only syntax.

### Removed

- The interactive terminal UI (`kavier-ui`). The `kavier` CLI and the Python API cover the same
  simulators.
- Docker image publishing from releases. The `Dockerfile` remains and CI builds it.
- The `dev/` analysis scripts.
- The Codecov upload and the README badges for Codecov and CI.

### Fixed

- `kavier energy` and other parquet readers could abort at exit with "terminate called without an
  active exception" (apache/arrow#34314); parquet files are now read by Arrow on one thread.
- Cluster simulator: a crash under `distributed-fcfs` when a zero-duration job tied with a later job;
  blank `nodes` cells in DataFrame input; zero-GPU replicas counted as hosted; infinite, NaN or negative
  job values; `plot_timeline` changing the matplotlib backend; tuple rows ignoring power and job id.
- Inference: prefix-cache token lists now load whenever `input_tokens` is present; fragments add up to the
  task duration for any export rate; `_sim_results.txt` is written as UTF-8; parquet traces with list
  token columns; invalid `prefix_policy` values are rejected, and `kv_cache` accepts booleans, 0 and 1,
  and on/off/true/false; `gpu_hour_price=0` gives a cost of 0; verbs keep the input DataFrame's index;
  small export rates show in the run summary.
- Training: enum method values get their interaction scale; zero epochs or tokens are no longer dropped;
  negative GPU, node and token counts are rejected; `calibrate --models ... --write` no longer overwrites
  the shipped tables; the multi-GPU gap fill is anchored at 1 GPU; NaN GPU counts in invalid rows.
- CLI: input errors, including a CSV without a required column, are reported as usage errors instead of
  tracebacks, and CSVs saved with a UTF-8 byte-order mark are read.
- `UnknownSpecError` can be pickled, so it reaches the caller from a process pool.

## [0.5.2] - 2026-09-19

`src/` is identical to `v0.5.1-thesis`, so 0.5.2 computes the same results as the thesis code.

### Added

- Test matrix on macOS and Linux, with `fail-fast: false`.
- Coverage with `pytest-cov` and an OIDC upload to Codecov that does not fail the build.
- `.pre-commit-config.yaml` with ruff, `ruff-format`, `end-of-file-fixer`, and `trailing-whitespace`.
- `.github/dependabot.yml` with weekly grouped updates for GitHub Actions and uv.
- `CITATION.cff` with the thesis as preferred citation, and `.zenodo.json` for archived releases.
- PEP 740 attestations on the PyPI publish step.
- This changelog.

### Changed

- The release workflow triggers on plain semver tags only (`v[0-9]+.[0-9]+.[0-9]+`).
- `requires-python` lowered from `>=3.13` to `>=3.11`, so IBM's `ado` (Python 3.11) can depend on Kavier.

### Fixed

- Missing final newline in `LICENSE.txt`, `.dockerignore`, and `docs/cluster-usage.md`.

## [0.5.1] - 2026-09-16

The code used for the MSc thesis experiments, tagged `v0.5.1-thesis`; the tag does not trigger a release.

### Added

- MFU in the training model and two goodput measures in the cluster simulator.
- Cluster policies `distributed-fcfs`, `distributed-backfill`, `consolidated-fcfs` (default), and `consolidated-backfill`.
- Cluster CLI options `--num-nodes`, `--node-gpus`, and `--out-nodes`, plus per-job placement columns.
- `docker-build` CI job for the `cli` and `ui` image targets.
- Cluster documentation in `docs/cluster-usage.md`.

### Changed

- Cluster policies renamed to the `distributed-` / `consolidated-` form; the default is `consolidated-fcfs`.
- Code cleanup: dead code removed, enums added, docstrings rewritten. No behaviour change.

[Unreleased]: https://github.com/atlarge-research/kavier/compare/v0.5.3...HEAD
[0.5.3]: https://github.com/atlarge-research/kavier/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/atlarge-research/kavier/compare/v0.5.1-thesis...v0.5.2
[0.5.1]: https://github.com/atlarge-research/kavier/releases/tag/v0.5.1-thesis
