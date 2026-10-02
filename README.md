# Kavier

Simulating the performance, sustainability, and efficiency of LLM ecosystems under inference and training.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/atlarge-research/Kavier/blob/master/LICENSE.txt)
[![Documentation](https://img.shields.io/badge/docs-site-green.svg)](https://atlarge-research.github.io/Kavier/)

Kavier is a physics-driven simulator. It predicts inference latency, training throughput, GPU utilization
and MFU, energy, carbon emissions, and cost per token or sample.

## Install

```bash
pip install kavier    # from PyPI
uv add kavier         # into a uv project
```

Python 3.11 or newer.

## Quick start

From a clone of this repository:

```bash
uv sync
uv run kavier inference --trace src/kavier/sdk/inference/data/input/input_example.csv
uv run kavier --help
```

The subcommands are `inference`, `training`, `cluster`, `energy`, `carbon`, and `calibrate` (needs the
`[calibration]` extra). Each documents its
flags with `--help`.

## Documentation

<https://atlarge-research.github.io/Kavier/>. Build it locally with `uv run --group docs mkdocs serve`.

## Development

`uv sync` installs the dev tools. CI runs these gates on every pull request
([ci.yml](https://github.com/atlarge-research/Kavier/blob/master/.github/workflows/ci.yml)):

```bash
uv run pre-commit run --all-files --show-diff-on-failure   # ruff check, ruff format, whitespace
uv run mypy --strict --show-error-codes \
  -p kavier.cli -p kavier.sdk.co2 -p kavier.sdk.cluster
uv run mypy --strict --follow-imports=skip --show-error-codes \
  src/kavier/__init__.py src/kavier/__main__.py \
  src/kavier/sdk/training/calibration/__init__.py \
  src/kavier/sdk/training/core/engine.py
uv run pytest --cov
uv run --group docs mkdocs build --strict
```

The calibration and plot tests are skipped unless you sync with `--extra calibration --extra plot`.
Run `uv run pre-commit install` once per clone to get the ruff and whitespace hooks on commit.

## Citation

See [CITATION.cff](https://github.com/atlarge-research/Kavier/blob/master/CITATION.cff).

## License

MIT. See [LICENSE.txt](https://github.com/atlarge-research/Kavier/blob/master/LICENSE.txt).
