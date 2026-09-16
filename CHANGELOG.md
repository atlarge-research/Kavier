# Changelog

All notable changes to Kavier are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Kavier adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The version lives only in
`pyproject.toml`; release tags mirror it as `vX.Y.Z`.

## [Unreleased]

### Added

- Test matrix now covers **macOS** as well as Linux (`ubuntu-latest`, `macos-latest` × Python 3.13,
  3.14), with `fail-fast: false` so one platform's failure cannot hide the other's. `wheel-smoke`
  and `docker-build` stay Linux-only — they exercise the shipped Linux artifacts.
- **Coverage reporting**: `pytest-cov` in the dev group, coverage settings in `pyproject.toml`
  (`uv run pytest --cov`), and a per-leg upload to Codecov via `codecov/codecov-action@v5` over
  OIDC — no upload token required. The upload is not a gate (`fail_ci_if_error: false`).
- **`.pre-commit-config.yaml`** running ruff (`--fix`) and `ruff-format` pinned to the same
  `v0.15.15` the dev group pins, plus `end-of-file-fixer` and `trailing-whitespace`. CI runs
  `pre-commit run --all-files`, so the hooks and CI cannot drift.
- **`.github/dependabot.yml`**: weekly, grouped updates for the `github-actions` and `uv`
  ecosystems, with runtime and development dependency bumps split into separate PRs.
- **`CITATION.cff`** (validated against CFF schema 1.2.0) with a `preferred-citation` for the Kavier
  thesis, and **`.zenodo.json`** so an archived release is attributed correctly. The Zenodo file
  takes effect only once the GitHub–Zenodo integration is enabled for the repository.
- **PEP 740 attestations** on the PyPI publish step: the sdist and wheel are signed with the
  workflow's Sigstore identity, so PyPI can attest the artifact came from this repo at this tag.
- This changelog.

### Changed

- The release workflow now triggers on **plain semver tags only**
  (`v[0-9]+.[0-9]+.[0-9]+`). The previous `v*.*.*` glob also matched suffixed marker tags, so a
  freeze tag such as `v0.5.1-thesis` would have started a real publish run.

### Fixed

- Missing final newline in `LICENSE.txt`, `.dockerignore` and `docs/cluster-usage.md`.

## [0.5.1] - 2026-09-16

The thesis freeze: the state of Kavier used for the MSc thesis experiments. Tagged
`v0.5.1-thesis` — a marker tag, deliberately outside the release trigger, so archiving the freeze
never publishes it.

### Added

- **MFU and scheduling-goodput prediction.** The training model reports Model FLOPs Utilization
  over the total GPU count across nodes, and the cluster simulator reports two distinct goodput
  measures: `goodput_jobs_per_s` (scheduling throughput) and `scheduling_goodput`
  (`Σ runtime_s / Σ turnaround_s`, scheduling efficiency).
- **Node-aware cluster scheduling.** Four policies spanning discipline × placement:
  `distributed-fcfs`, `distributed-backfill`, `consolidated-fcfs` (default) and
  `consolidated-backfill`. The consolidated modes honour each job's `nodes` column, placing a job's
  replicas on exactly that many distinct co-located nodes rather than fragmenting them.
- **Cluster CLI and outputs**: `--num-nodes` / `--node-gpus`, a per-node CSV via `--out-nodes`
  (utilisation, jobs hosted, peak GPUs, idle time, energy), a `node:gpus` column and a
  human-readable `placement` column in the per-job CSV, and a `node_activity()` peak/idle helper.
- **`docker-build` CI job** building both the `cli` and `ui` image targets on every push, so a
  broken Dockerfile stage fails CI rather than a release.
- **Cluster documentation** (`docs/cluster-usage.md`) covering trace columns, policies and the
  per-job/per-cluster metrics, plus the goodput and MFU analysis scripts behind the
  scheduling-efficiency numbers.

### Changed

- Cluster policies renamed to the `distributed-` / `consolidated-` form; the default is now
  `consolidated-fcfs`.
- Codebase hygiene pass: dead code removed, enums introduced, units and defaults homed in one
  place, docstrings rewritten and long functions split. No behaviour change.

[Unreleased]: https://github.com/atlarge-research/kavier/compare/v0.5.1-thesis...HEAD
[0.5.1]: https://github.com/atlarge-research/kavier/releases/tag/v0.5.1-thesis
