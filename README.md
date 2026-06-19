# Kavier

Simulating performance, sustainability, and efficiency of LLM Ecosystems under inference and training.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-main-green.svg)](docs/index.html)
[![CI](https://github.com/atlarge-research/kavier/actions/workflows/ci.yml/badge.svg)](https://github.com/atlarge-research/kavier/actions/workflows/ci.yml)

---

This repository is the home of Kavier, the first scientific instrument for
predicting performance, sustainability, and efficiency of LLM ecosystems under
inference and training, through discrete-event, cache-aware simulation.

Kavier helps operators, researchers, and engineers predict:
* **Performance** — inference latencies, training throughput, GPU utilization
* **Sustainability** — energy consumption, carbon emissions (kgCO2/Mtoken/s)
* **Financial efficiency** — cost per token/sample given GPU-hour prices

## Quick start

```bash
git clone https://github.com/atlarge-research/kavier.git
cd kavier

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
```

Run your first simulation against the tiny bundled synthetic example trace:

```bash
kavier-perf --trace src/kavier_inference/data/input/input_example.csv
```

Congrats! You have just run your first simulation with Kavier! 🎉

If you installed Kavier from PyPI (`pip install kavier`) you have no `src/`
directory; the same synthetic example trace ships inside the package, so resolve
its path via `importlib.resources`:

```bash
TRACE=$(python -c "from importlib.resources import files; print(files('kavier_inference')/'data/input/input_example.csv')")
kavier-perf --trace "$TRACE"
```

## Structure

Kavier is organized into the following first-party packages:

```
src/
├── kavier/              # Umbrella facade (re-exports the sub-packages)
├── kavier_inference/    # Inference simulation (kavier-perf)
├── kavier_training/     # Training simulation (kavier-train)
├── kavier_energy/       # Energy calculator (kavier-energy)
├── kavier_co2/          # Carbon emissions (kavier-co2)
├── kavier_library/      # Shared GPU & LLM specifications
├── kavier_opendc/       # OpenDC workload export (tasks/fragments)
├── kavier_io/           # Shared I/O utilities
└── tests/               # Test suites
```

## Documentation

See [docs/index.html](docs/index.html) for the main documentation: getting started,
the Kavier CLI (`kavier-perf`, `kavier-train`, `kavier-energy`, `kavier-co2`),
structure, and the contributing guide.

## Contributing

Questions, suggestions and contributions are welcome and appreciated!
Please refer to the [contributing guide](docs/contributing.md) for more details.

## License

Kavier is distributed under the MIT license. See [LICENSE.txt](/LICENSE.txt).
