# Contributing to Kavier or OpenDC (or both!)

First of all, thank you for wanting to contribute to our project(s)!

You can contribute in various meaningful ways:

- Report a bug.
- Propose new functionality for using this project.
- Contribute improvements to the code and documentation.
- Provide feedback about how we can improve the project.
- Help answer questions on our
  [Discussions](https://github.com/atlarge-research/kavier/discussions) page.

## Let's get in touch! {#touch}

If you have any questions, suggestions, or just want to chat about the project, feel free to reach
out to us via:

- `mail@radu-nicolae.com`
- `info@atlarge-research.com`

!!! tip

    CC both addresses! :D

## Want to report a bug or suggest a feature? {#bug}

Please file an issue! First, have a look if someone has already filed an issue addressing your
concern. If there already is such an issue, feel free to comment on the issue to show your support
for it, or to add additional information that might be helpful. You can also just react with a
thumbs-up to the issue, to indicate that you'd be interested in its resolution. This can help us
prioritize what we spend our development time on.

If you can't find an issue that fits your problem or feature request, open a new one. Describe actual
and expected behavior, and be as detailed as you can. We'll get back to you asap!

## Want to contribute code? {#code}

That's great! If you want to contribute to this repository,
[fork it](https://github.com/atlarge-research/kavier/fork) and submit a pull request here when
you're ready! Be sure to describe *what* you changed and *why* you changed it, to help us understand
what your contribution is about.

A quick note on commit messages: Please follow common Git standards when writing commit messages,
see [this post](https://cbea.ms/git-commit/) for details.

## Set up a development environment {#dev-env}

Kavier is a standard `src/`-layout Python project (Python >= **3.13**), developed with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/atlarge-research/kavier.git
cd kavier
uv sync                  # creates .venv and installs kavier plus pytest, hypothesis, ruff, mypy
```

Add the `calibration` extra (`uv sync --extra calibration`) to run the scipy/scikit-learn
calibration-refit tests; without it, `test_engine_regen.py` and friends `importorskip`-skip.

## Checks to run before you push {#checks}

CI ([.github/workflows/ci.yml](https://github.com/atlarge-research/kavier/blob/master/.github/workflows/ci.yml))
runs exactly these on every push and pull request (tests on Python 3.13 and 3.14), so run them
locally first:

```bash
uv run pytest                 # full test suite
uv run ruff check .           # lint
uv run ruff format --check .  # formatting (CI pins ruff==0.15.15; fix with: uv run ruff format .)

# Strict typing is gated INCREMENTALLY (the full tree is not strict-clean yet):
uv run mypy --strict -p kavier.cli -p kavier.ui -p kavier.sdk.co2
uv run mypy --strict --follow-imports=skip \
  src/kavier/__init__.py src/kavier/__main__.py \
  src/kavier/sdk/training/calibration/__init__.py \
  src/kavier/sdk/training/core/engine.py
```

CI also smoke-tests that the CLI entry points resolve (`kavier inference --help`,
`kavier training --help`, `kavier energy --help`, `kavier carbon --help`).

## Where the code lives {#where}

Everything lives under `src/kavier/`, in three top-level parts — `cli/` (the unified `kavier`
command), `ui/` (the `kavier-ui` REPL), and `sdk/` (all the modelling). There is **no import hook**
and there are **no top-level `kavier_*` packages**.

The public API verbs (`performance` / `energy` / `efficiency` / `carbon`) are implemented in
`src/kavier/sdk/inference/facade.py` and `src/kavier/sdk/training/facade.py`, and re-exported
**lazily** from each `sdk` subpackage. `kavier.inference` / `kavier.training` are convenience
**aliases** for `kavier.sdk.inference` / `kavier.sdk.training`, so
`import kavier; kavier.inference.performance(batch)` resolves straight into the facade. The rest of
each `kavier.sdk.*` package holds the engines, calibration, and the per-domain CLIs the unified
command dispatches to. See the [Architecture page](architecture.md) for the full map.

## Looking for a first contribution? {#first}

Add a GPU to the spec library in `src/kavier/sdk/library/gpu.py` and watch the parametrized tests
pick it up. The [Library component page](library.md#contribute) walks through it step by step, and
the README's
["Your first change"](https://github.com/atlarge-research/kavier/blob/master/README.md#your-first-change)
section mirrors it.

---

Kavier is distributed under the MIT license. See
[LICENSE.txt](https://github.com/atlarge-research/kavier/blob/master/LICENSE.txt).
