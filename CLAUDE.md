# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Kavier is a **predictive simulator for LLM ecosystems**: describe a workload (model + GPU + request/job
sizes) and it estimates inference latency/throughput, training throughput/runtime, GPU utilization, energy
(Wh), carbon (gCO₂), and cost ($/Mtoken) — analytically, without running the real hardware. It is a research
instrument (AtLarge Research), ~5k lines of source under `src/`, distributed on PyPI as `kavier`.

## Commands

Environment: the project is uv-native (`build-backend = "uv_build"`, committed `uv.lock`). Set up with
`uv sync` (add `--extra calibration` to run the scipy/scikit-learn calibration-refit tests; without it,
`test_engine_regen.py` and friends `importorskip`-skip). Dev tools live in the PEP 735 `dev` group,
synced by default (so `pip install .[dev]` no longer works — use uv).

```bash
# Tests (mirrors CI). Tests live at repo-root tests/; testpaths is set in pyproject.toml.
uv run pytest
uv run pytest tests/test_co2/test_emissions.py -k irregular      # single file / -k filter
uv run pytest tests/test_co2/test_emissions.py::test_multiple_fragments_same_window_accumulate

# Lint + format (CI pins ruff==0.15.15; a version drift can change formatting)
uv run ruff check .
uv run ruff format --check .    # fix with: uv run ruff format .

# Typing — strict mypy is gated INCREMENTALLY (not full-tree; see ci.yml for the burn-down list)
uv run mypy --strict --show-error-codes -p kavier.cli -p kavier.ui -p kavier.sdk.co2
uv run mypy --strict --follow-imports=skip --show-error-codes \
  src/kavier/__init__.py src/kavier/__main__.py \
  src/kavier/sdk/training/calibration/__init__.py src/kavier/sdk/training/core/engine.py
```

Run the tool: one CLI, five subcommands — `uv run kavier inference --trace
src/kavier/sdk/inference/data/input/input_example.csv`, `kavier training`, `kavier energy`,
`kavier carbon`, and `kavier calibrate <profiling.csv>` (fit a training-calibration table; needs the
`[calibration]` extra); plus the interactive REPL `uv run kavier-ui` (POSIX-only — termios/tty). Both
entrypoints also run as modules: `python -m kavier` (CLI), `python -m kavier.ui`. Programmatic:
`kavier.inference.performance(batch)` / `kavier.training.performance(batch)` etc., where a batch is a
DataFrame, `list[dict]`, or single `dict` and the return is input rows + predicted columns.

## Architecture — the big picture

**One package, layered.** Everything is under `src/kavier/`, whose top level is exactly `cli/` (the
unified `kavier` command), `ui/` (the REPL), and `sdk/` (the functionality) — no import hook, no
`kavier_<sub>` top-level packages (removed in 0.5.0). Each `kavier.sdk.<domain>` package IS that
functionality: the engine (`core/`, `stages/`, `calibration/`, …) plus, for inference/training, the
public batch-predictor verbs (`performance`/`energy`/`efficiency`/`carbon`) in `facade.py`, re-exported
**lazily** from the package `__init__`. `kavier.inference` / `kavier.training` are convenience aliases
for `kavier.sdk.inference` / `kavier.sdk.training`. `kavier/__init__.py` is **lazy** (PEP 562
`__getattr__`) so a bare `import kavier` stays cheap and stdlib-light. The engines:

- `sdk/inference/` — discrete per-request inference sim (`kavier inference`); verbs in `facade.py`,
  `core/runner.py::simulate_one` drives the `stages/` physics.
- `sdk/training/` — analytical training model (`kavier training`) in `core/engine.py`, wrapped by the
  `calibration/` correction layer; verbs in `facade.py`.
- `sdk/energy/` — `kavier energy` subcommand + the shared GPU power model (`engine.py::mse_power`).
- `sdk/co2/` — carbon (`kavier carbon`): `Fragment` (power×time) integrated over a `CarbonTrace` (grid intensity).
- `sdk/library/` — shared **static** GPU/LLM specs (the domain entities); `lookup.py` resolves names.
- `sdk/io/` — trace parsing (`input_spec.py`), YAML config loading (`config.py`), and OpenDC parquet
  export (`opendc/`).

**CLI vs SDK — clean split.** ALL argument parsing and command glue lives in `kavier/cli/`: `main.py`
(the argparse dispatcher) + one module per subcommand (`inference.py`/`training.py`/`energy.py`/
`carbon.py`/`calibrate.py`), plus shared `_shared.py` (FriendlyParser + `--config` folding) and
`_args.py` (training arg-builders). Each subcommand parses its args and *calls into* the SDK (engines/services/facades). The
SDK is a pure library with NO argparse. `ui/sims.py` adapts the facades (no modelling logic of its own).

**Data flow.** A workload row → engine → per-request/per-step metrics (+ OpenDC `tasks`/`fragments`) →
the facades chain that into carbon (`compute_emissions`) and energy/cost. Tuning constants for the
inference physics live in `kavier/sdk/io/constants.py` (`COMPUTE_EFFICIENCY`, `MEMORY_EFFICIENCY`, …).

**The prediction models (know these before changing formulas):**
- *Inference* is a per-request **roofline**: prefill is compute-bound, decode is memory-bandwidth-bound
  (`stages/decode.py`), linear with KV cache and **quadratic (`n(n+1)/2`) without it**. It is
  **single-stream — there is NO batching/queueing/contention**; "throughput" is `total_tokens / Σ per-request
  time`, so identical requests give identical p50==p95. FLOPs/weights use `active_params` (MoE-aware).
- *Training* (`core/engine.py`) is a first-principles **6ND FLOPs + Adam-memory + ring-all-reduce** step time,
  then throughput is multiplied/divided by fitted **calibration** factors (see contract below).
- *Power* is `mse_power` = `idle + (max−idle)·(2u − uʳ)`; every shipped GPU has `r = mse_calib_factor = 1.0`,
  so **in practice it's a linear ramp**. **Inference bills GPU max power (TDP); training bills calibrated
  `mse_power`** — same-looking energy numbers, different bases.

## Critical contracts & gotchas

- **Calibration is keyed on EXACT catalog names, and the two engines use different name conventions.**
  Training is calibrated only for a handful of models/GPUs whose keys are the *full* names
  (`granite-3-8b`, `NVIDIA-A100-SXM4-80GB`, …) in `calibration/calibration.json`. Passing an unfit name —
  including the *short* inference keys (`Llama-3-8B`, `A100-80GB`) or the default `Llama-3-8B` — **silently
  falls back to neutral 1.0 with a one-time `UserWarning`**, giving raw uncalibrated physics. Getting a
  number ≠ getting a calibrated number.
- **Calibration ships as per-use-case, selectable version files.** The runtime accessor
  (`use_calibration("<name>")`, listed by `available_calibrations()`, or the `KAVIER_CALIBRATION` env var)
  picks one table: the default `calibration.json` and `6model` (`versions/calibration_6model.json`) are the
  thesis *exploration* fits (6 LLMs, byte-identical); `4model` (`versions/calibration_4model.json`) is the
  thesis *validation* experiment (4 LLMs); `allmodels` (`versions/calibration_allmodels.json`) is the
  NON-thesis fit of all ~32 calibratable catalog models (any-GPU-count curation). To fit a table from an
  arbitrary profiling CSV, `kavier calibrate <profiling.csv> [--output PATH] [--models a,b,…]` runs the
  from-scratch two-tier fit (the backend for Coastline's `coastline-tune --method kavier`); unlike shipped
  `regenerate()` it curates at ANY GPU count (no ≤8 cap) and does NOT reproduce `calibration.json`
  byte-for-byte.
- **`kavier/sdk/training/calibration/__init__.py` must stay stdlib-only** (import-light contract):
  importing it must not pull scipy/sklearn/numpy/pandas. This is why `kavier/__init__.py`,
  `kavier/sdk/__init__.py`, and `kavier/sdk/training/__init__.py` MUST stay import-light — a bare
  `import kavier.sdk.training.calibration` executes all of them. The heavy fitting engine is
  `calibration/engine.py` (dev-only, needs the `[calibration]` extra; `calibration_override(...)` lives
  there). The accessors read a module global `_CAL` swapped live by callers, and both import spellings
  (`kavier.sdk.training.calibration` and `kavier.sdk.training.core.calibration`) must resolve to **one**
  module object — `test_public_surface.py` / `test_kavier_namespace.py` enforce this.
- **Spec objects (`GPUSpec`/`LLMSpec`) do NO validation.** Invariants (positive fields, `idle ≤ max`,
  `key == name`, `0 < active_params ≤ m_params`, unit conversions) are enforced entirely by
  `test_library/test_spec_consistency.py`, parametrized over the whole catalog. Lookups are exact-match,
  case-sensitive, **no aliases** (`A100-80GB` and `NVIDIA-A100-80GB-PCIe` are distinct keys).
- **Two divergent energy/carbon paths that share no code**: the self-contained facade estimate (flat
  synthetic trace) vs. the OpenDC-readback path (`kavier energy` sums an external `powerSource.parquet`).
  They can disagree for the "same" run — this is a known, documented limitation, not a bug.
- **The two "utilization/goodput" pairs are near-homonyms — don't conflate them.** Training `performance`
  emits both `gpu_compute_utilization` (the *assumed* physical MFU the engine feeds INTO the throughput
  prediction) and `mean_flops_utilization` (the *realized* 6N·tok/s÷(gpus·peak) MFU backed OUT of the
  predicted throughput). They share the catalog peak basis but are NOT ordered — a fitted `method_scale > 1`
  (lora/gptq-lora) can push realized above assumed. Predicted MFU is self-consistent, not independently
  validated (it circularly reflects `mfu_factor`); on Hopper it reads against NVIDIA's *with-sparsity* peak,
  so ~2× lower than the dense-peak `mfu_calculator.py`. Separately, the cluster sim has `goodput_jobs_per_s`
  (throughput, jobs/s) AND `scheduling_goodput` (efficiency, Σruntime÷Σturnaround ∈ [0,1], over *scheduled*
  jobs only — dropped jobs are excluded). Predicted `scheduling_goodput` inherits the fidelity of the input
  job durations (Coastline feeds Kavier-predicted durations into the sim).
- **Calibration regeneration is deterministic.** `calibration/engine.py --check` (and
  `test_engine_regen.py`) rebuild the JSON from scratch (seed-42 Powell); the fit is brittle to
  profiling-CSV row order. Because adding catalog models shifts the >8-GPU mgc (see next gotcha), the
  guard now asserts byte-for-byte equality only on the ≤8 (used) values and structural equality on the
  >8 entries. Don't hand-edit the shipped tables — the thesis files are frozen; fit new ones with
  `kavier calibrate`.
- **The >8-GPU `multi_gpu_correction` is coupled to the catalog.** The shipped tables' large-count entries
  (16/32/64/128) are a single global median over EVERY catalog model's big-GPU runs, so they shift as the
  catalog grows. The recommender only uses ≤8-GPU predictions, so the byte-identity guard now pins the frozen
  thesis files only on their ≤8 (used) values — the >8 entries are allowed to move. P1 finding: `calibrate`'s
  uncapped path makes `comm_scale` data-identifiable (~1.0) vs. the ≤8-only fit's regularization-pinned
  ~1.23, and accuracy degrades at high GPU counts (128-GPU MdAPE >30%). Treat >8-GPU as out-of-regime.

## Testing conventions

Tests are layered: **invariant** tests assert physical laws (monotonicity, non-negativity, roofline
scaling, `power ∈ [idle,max]`, fragments tile duration) parametrized over the *entire* GPU×LLM library;
**regression** tests cite the commit/issue they guard and hand-derive expected values (grep for issue
numbers); **property-based** tests use Hypothesis; **integration** tests shell out to the real CLI via
`python -m kavier.cli <subcommand>`.
Some accuracy/determinism tests skip on a clean checkout (they need the `[calibration]` extra or an
**unshipped** internal validation CSV) — a skip there is expected, not a failure.

## Where the model/limitations are documented

The documentation is a MkDocs (Material) site under `docs/` (`docs/mkdocs.yml`, pages in
`docs/content/`; build with `cd docs && mkdocs build`). The per-component pages
(`docs/content/performance.md`, `docs/content/energy.md`, `docs/content/co2.md`,
`docs/content/efficiency.md`, `docs/content/library.md`) state the intended math for every engine in
their Formulas sections (with paper citations). `docs/content/known-weaknesses.md` enumerates the
project's own admitted limits (e.g. inference has no measured-accuracy validation; most models run
uncalibrated; MoE/scaling caveats) — read it before trusting or "improving" a number, and keep it in sync
when you change modelling behaviour. Only the training model has a published accuracy figure, and it is
validated only on internal data for its calibrated model/GPU set; treat all outputs as planning estimates.
