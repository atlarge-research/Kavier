# Kavier

Simulating performance, sustainability, and efficiency of LLM Ecosystems under inference and training.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-main-green.svg)](docs)
[![CI](https://github.com/atlarge-research/kavier/actions/workflows/ci.yml/badge.svg)](https://github.com/atlarge-research/kavier/actions/workflows/ci.yml)

---

This repository is the home of Kavier, the first scientific instrument for
predicting performance, sustainability, and efficiency of LLM ecosystems under
inference and training, through discrete-event, cache-aware simulation.

Kavier helps operators, researchers, and engineers predict:
* **Performance** — inference latencies, training throughput, GPU utilization
* **Sustainability** — energy consumption, carbon emissions (kgCO2/Mtoken/s)
* **Financial efficiency** — cost per token/sample given GPU-hour prices

## Structure

Kavier is organized into 5 main components:

```
src/
├── kavier_inference/    # Inference simulation (kavier-perf, kavier-eff)
├── kavier_training/     # Training simulation (kavier-train)
├── library/             # Shared GPU & LLM specifications
├── io/                  # Shared I/O utilities
└── tests/               # Test suites
```

## Documentation

We divide the documentation into the following sections:

* [Getting Started](docs/getting-started.md)
* Using the Kavier CLI
    * [Kavier Performance CLI](docs/cli-performance.md) - `kavier-perf`
    * [Kavier Efficiency CLI](docs/cli-efficiency.md) - `kavier-eff`
    * Kavier Training CLI - `kavier-train`
* [Restructuring Summary](RESTRUCTURE_SUMMARY.md) - v0.2.0 architecture changes
* Thesis
* [Contributing guide](docs/contributing.md)

## Contributing

Questions, suggestions and contributions are welcome and appreciated!
Please refer to the [contributing guidelines](CONTRIBUTING.md) for more details.

## License

Kavier is distributed under the MIT license. See [LICENSE.txt](/LICENSE.txt).

