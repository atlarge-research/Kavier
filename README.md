# Kavier

Simulating performance, sustainability, and efficiency of LLM Ecosystems under inference and training.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-main-green.svg)](docs/content/index.md)
[![CI](https://github.com/atlarge-research/kavier/actions/workflows/ci.yml/badge.svg)](https://github.com/atlarge-research/kavier/actions/workflows/ci.yml)

---

This repository is the home of Kavier, the first scientific instrument for
predicting performance, sustainability, and efficiency of LLM ecosystems under
inference and training.

Kavier helps operators, researchers, and engineers predict:
* **Performance** — inference latencies, training throughput, GPU utilization
* **Sustainability** — energy consumption, carbon emissions (gCO2/Mtoken)
* **Efficiency** — financial and energy cost per token/sample given GPU-hour prices

## Quick start

Kavier is developed with [uv](https://docs.astral.sh/uv/). Install uv, then:

```bash
git clone https://github.com/atlarge-research/kavier.git
cd kavier

uv sync            # create .venv and install Kavier + dev tools from uv.lock
```

Run your first simulation against the tiny bundled synthetic example trace:

```bash
uv run kavier inference --trace src/kavier/sdk/inference/data/input/input_example.csv
```

Congrats! You have just run your first simulation with Kavier! 🎉

Kavier is one CLI with several subcommands — `inference`, `training`, `cluster`, `energy`, `carbon`:

```bash
uv run kavier --help
uv run kavier training --help
```

Or launch the **interactive UI** and pick a simulator, model and GPU from guided
menus, then chain into energy/carbon or export OpenDC:

```bash
uv run kavier-ui
```

> The interactive UI is POSIX-only (Linux/macOS — it drives the terminal via `termios`);
> on Windows, use the one-shot CLI (`kavier inference`, `kavier training`, …) instead.

Both entrypoints are also reachable as modules: `python -m kavier` (CLI) and `python -m kavier.ui`.

If you installed Kavier from PyPI (`pip install kavier`) you have no `src/`
directory; the same synthetic example trace ships inside the package, so resolve
its path via `importlib.resources`:

```bash
TRACE=$(python -c "from importlib.resources import files; print(files('kavier.sdk.inference')/'data/input/input_example.csv')")
kavier inference --trace "$TRACE"
```

## Simulating a cluster

`kavier cluster` is a FIFO/backfill **queuing simulator**: give it a CSV trace of jobs (arrival
time, GPUs requested, GPU-locked duration) and a fixed cluster size, and it schedules them and
reports per-job timings (wait, start/end, runtime, energy) plus cluster metrics (makespan,
utilization, goodput, peak queue). A tiny example trace ships with Kavier — swap in your own:

```bash
# trace columns: submit_s,gpus,duration_s[,nodes,power_w_per_gpu]
uv run kavier cluster --jobs src/kavier/sdk/cluster/data/input/trace_example.csv \
  --policy fcfs --num-gpus 32 --out per_job.csv
```

Per-job results go to `--out` (CSV); the cluster summary prints as JSON. Programmatic use:
`from kavier.sdk.cluster import schedule` — call `schedule(pd.read_csv("trace.csv"), policy="fcfs",
num_gpus=32)` and read `result.jobs` / `result.cluster`.

## Structure

Kavier is a single importable package, `kavier`:

```
src/kavier/
├── cli/              # the unified `kavier` CLI (subcommands: inference/training/cluster/energy/carbon)
├── ui/               # the interactive REPL (the `kavier-ui` command)
└── sdk/              # the functionality — one subpackage per domain
    ├── inference/    # per-request inference simulator + the verb facade (facade.py)
    ├── training/     # analytical training model + calibration + the verb facade (facade.py)
    ├── cluster/      # FIFO/backfill cluster queuing simulator + the schedule() facade
    ├── energy/       # GPU power / efficiency
    ├── co2/          # carbon emissions
    ├── io/           # trace I/O + OpenDC export (io/opendc/)
    └── library/      # shared GPU & LLM specifications
tests/                # test suites (run with `uv run pytest`)
```

The layout is layered:

* **Public verbs** — `kavier.inference.performance / energy / efficiency / carbon` (and the training
  equivalents) are the batch predictors; each takes a workload batch (DataFrame, `list[dict]`, or a
  single `dict`) and returns the input rows plus predicted columns. `kavier.inference` /
  `kavier.training` are convenience aliases for `kavier.sdk.inference` / `kavier.sdk.training`, where
  the verbs live (in `facade.py`). `import kavier` is lazy, so it stays cheap until you touch a verb.
* **Engines** — each `kavier.sdk.*` package holds the actual simulators, calibration, and specs. It's a
  pure library: all argument parsing lives in `kavier.cli` (one module per subcommand), which calls into
  these engines. The unified `kavier` command dispatches to those `cli/` modules.

## Development

`uv sync` (from [Quick start](#quick-start)) installs the project plus the dev tools (pytest, ruff,
mypy). Before pushing, run the same gates CI enforces on every push/PR
([.github/workflows/ci.yml](.github/workflows/ci.yml)):

```bash
uv run pytest                 # test suite
uv run ruff check .           # lint
uv run ruff format --check .  # formatting (CI pins ruff==0.15.15; fix with: uv run ruff format .)

# Strict typing is gated incrementally (the full tree is not strict-clean yet):
uv run mypy --strict -p kavier.cli -p kavier.ui -p kavier.sdk.co2
uv run mypy --strict --follow-imports=skip \
  src/kavier/__init__.py src/kavier/__main__.py \
  src/kavier/sdk/training/calibration/__init__.py \
  src/kavier/sdk/training/core/engine.py
```

Add the `calibration` extra (`uv sync --extra calibration`) to run the scipy/scikit-learn
calibration-refit tests; without it, `test_engine_regen.py` and friends `importorskip`-skip.

A multi-stage [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml) are provided for
containerized runs (they build a wheel and install it — no source tree, no `PYTHONPATH`), but Docker
is optional — every tutorial and the workflow above use `uv`:

```bash
docker build --target cli -t kavier:cli .   # the unified CLI
docker run --rm kavier:cli inference --help
docker build --target ui -t kavier:ui .     # the interactive REPL
docker run --rm -it kavier:ui
```

### Your first change

A good starter task: add a GPU to the built-in spec library.

1. Open [src/kavier/sdk/library/gpu.py](src/kavier/sdk/library/gpu.py) — `GPU_SPEC_LIBRARY` is a plain
   dict of `GPUSpec` entries. Copy an existing entry, tweak the numbers, and make the dict key
   match the `gpu_name` field.
2. Run the library tests — they parametrize over every entry, so your GPU is validated
   automatically (positive specs, `idle_power_w <= max_power_w`, key == name, unit conversions):

   ```bash
   uv run pytest tests/test_library
   ```

3. Simulate on your new GPU with the bundled example trace:

   ```bash
   uv run kavier inference --gpu "YourGPU" --trace src/kavier/sdk/inference/data/input/input_example.csv
   ```

Finish with the full gate set above, then open a PR — see the
[contributing guide](docs/content/contributing.md).

## Documentation

The documentation is a [MkDocs](https://www.mkdocs.org/) (Material) site under
[`docs/`](docs/). Start reading at [`docs/content/index.md`](docs/content/index.md): getting
started, the Kavier CLI (`kavier inference`, `kavier training`, `kavier energy`, `kavier carbon`)
and the `kavier-ui` interactive UI, the per-component pages (performance, energy, CO2, efficiency,
library), and the contributing guide.

Build and serve it locally:

```bash
cd docs
pip install -r requirements.txt
mkdocs serve            # live site on http://localhost:8000
mkdocs build            # or emit the static site to docs/site/
```

Or with Docker (documentation only):

```bash
docker build -t kavier-docs docs/
docker run --rm -p 8000:8000 kavier-docs
```

## Contributing

Questions, suggestions and contributions are welcome and appreciated!
Please refer to the [contributing guide](docs/content/contributing.md) for more details.

## License

Kavier is distributed under the MIT license. See [LICENSE.txt](/LICENSE.txt).
