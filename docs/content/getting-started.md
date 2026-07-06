# Getting started

Install Kavier, run the CLI, call the Python SDK, and launch the interactive UI — in under 10
minutes.

## 1. Pre-requisites {#prereqs}

1. Python >= **3.13** (required by `pyproject.toml`; CI tests 3.13 and 3.14).
2. [**uv**](https://docs.astral.sh/uv/) — the project's package/venv manager. A committed
   `uv.lock` pins every dependency.

!!! tip "Tip for Apple Silicon"

    `brew install python@3.13`, then `brew install uv` (or the `curl` installer from the uv docs).
    uv can also fetch a suitable Python for you on first `uv sync`.

## 2. Clone & sync the environment {#venv}

```bash
git clone https://github.com/atlarge-research/kavier.git
cd kavier

uv sync   # creates .venv and installs Kavier + dev tools from uv.lock
```

Add the `calibration` extra (`uv sync --extra calibration`) only if you plan to run the
scipy/scikit-learn calibration-refit tests; without it, those tests `importorskip`-skip.

## 3. Verify the install {#install}

```bash
uv run kavier --help
```

`uv sync` resolved runtime deps (pandas, pyarrow, numpy, …), registered the editable package so
edits in `src/` apply instantly, and installed the entry points — **one CLI with four
subcommands**, plus the interactive UI:

- `kavier inference` — inference simulation
- `kavier training` — fine-tuning (training) prediction
- `kavier energy` — efficiency per million tokens
- `kavier carbon` — emissions against a carbon trace
- `kavier-ui` — the interactive REPL

Both entrypoints are also reachable as modules: `python -m kavier` (CLI) and `python -m kavier.ui`
(UI).

## 4. Run your first simulation (CLI) {#first-run}

### 4.1 The bundled trace

A tiny inference trace ships at `src/kavier/sdk/inference/data/input/input_example.csv`. Or bring
your own — the columns are:

- **[REQUIRED]** `num_input_tokens` — input tokens in the request
- **[REQUIRED]** `num_output_tokens` — output tokens in the response
- **[OPTIONAL]** `session_id` — session id (used by `--cache_scope session`)
- **[OPTIONAL]** `input_tokens` / `output_tokens` — actual token-id lists

Naming note: in trace CSVs, `input_tokens` is the token-id **list** and `num_input_tokens` is the
**count** — whereas in the Python SDK (`kavier.inference.*`), `input_tokens` is a scalar count.

!!! note "The large demo trace (local-only)"

    The bundled `input_example.csv` (~562 B, counts-only) is a tiny synthetic trace. The original
    kavier-inference demo trace — a ~74 MB file of Llama-tokenized chat token-IDs derived from
    public chat datasets (LMSYS-Chat-1M / WildChat) — is **not shipped in the wheel and not
    committed**: its size bloats the package and its dataset license/provenance is unresolved under
    Kavier's MIT license.

    It is preserved in git history. The packaging commit `26a5aac` removed it, so its parent still
    has the file:

    ```bash
    git show 26a5aac~1:src/kavier_inference/data/input/input_example.csv > input_example_large.csv
    ```

    (The path in that historical commit predates the 0.5.0 layout; the current bundled trace lives
    at `src/kavier/sdk/inference/data/input/input_example.csv`.) Confirm the exact source dataset
    and its license before any redistribution.

### 4.2 Your first simulation

```bash
uv run kavier inference --trace src/kavier/sdk/inference/data/input/input_example.csv
```

### 4.3 Your first output

A timestamped folder appears under `--output_folder` with two Parquet files, plus an
OpenDC-compatible export; a readable summary prints to your terminal.

- `tasks.parquet` — one row per request (latency, `total_tokens`, …)
- `fragments.parquet` — GPU-utilisation snapshots
- a **SIMULATION SUMMARY** table on **stdout** (prefill/decode time, p95 latency, cache hit ratio)

!!! success "Milestone"

    Congrats — you just ran your first Kavier inference simulation!

## 5. A typical inference run {#typical}

```bash
uv run kavier inference \
  --llm         Llama-3-8B \
  --gpu         A10 \
  --trace       src/kavier/sdk/inference/data/input/input_example.csv \
  --kv_cache    on \
  --export_rate 0.1 \
  --flush_size  1000
```

See the [Performance component](performance.md#inference) for all inference flags.

## 6. Predicting a fine-tuning job {#train}

Training is a **separate** analytical model — no trace, just a config:

```bash
uv run kavier training \
  --model_name mistral-7b-v0.1 --method lora \
  --gpu_model NVIDIA-A100-SXM4-80GB --tokens_per_sample 1024 \
  --batch_size 4 --number_gpus 8 --number_nodes 1
```

It prints a JSON object with tokens/sec, MFU and power. See the
[Performance component](performance.md#training).

## 7. The Python SDK {#sdk}

Everything the CLI does is available programmatically. `import kavier` exposes two workload
namespaces — `inference` and `training` — each with the same four predictors (`performance` /
`energy` / `efficiency` / `carbon`). Every verb takes a *batch* (a pandas DataFrame, a
`list[dict]`, or a single `dict` — one row per workload) and returns a DataFrame: your input rows
plus the predicted columns.

```python
import pandas as pd
import kavier

# In the SDK, input_tokens / output_tokens are scalar *counts* (one row per workload).
inference_batch = pd.DataFrame(
    [
        {"model": "Llama-3-8B", "gpu": "A10", "num_requests": 128, "input_tokens": 512, "output_tokens": 128},
    ]
)

performance = kavier.inference.performance(inference_batch)  # + p50_ms, throughput_tok_s, util
energy      = kavier.inference.energy(inference_batch)       # + energy_wh, energy_per_mtoken_wh
efficiency  = kavier.inference.efficiency(inference_batch)   # + financial_per_mtoken ($/Mtoken)
carbon      = kavier.inference.carbon(inference_batch)       # + carbon_per_mtoken_g (gCO2)

print(performance)
```

The training verbs work the same way. The canonical, test-exercised example is `docs/usage.py`,
reproduced in full here:

```python
--8<-- "usage.py"
```

Run it end-to-end with:

```bash
uv run python docs/usage.py
```

## 8. The interactive UI {#ui}

Prefer guided menus? Launch the REPL and pick a simulator, model, and GPU, then chain into
energy/carbon or export OpenDC:

```bash
uv run kavier-ui
```

!!! warning "POSIX only"

    The interactive UI drives the terminal via `termios`, so it runs on Linux/macOS only. On
    Windows, use the one-shot CLI (`kavier inference`, `kavier training`, …) or the SDK instead.

## 9. OpenDC sustainability {#opendc}

Kavier integrates with the peer-reviewed [OpenDC](https://opendc.org/) for sustainability. After an
inference run, feed the exported workload through OpenDC (see its docs) to produce a
`powerSource.parquet` — no manual conversion needed.

## 10. Post-processing: efficiency metrics {#efficiency}

With a `tasks.parquet` (from Kavier) and a `powerSource.parquet` (from OpenDC):

```bash
uv run kavier energy \
  --kavier kavier_output/<TIMESTAMP>/tasks.parquet \
  --opendc /path/to/powerSource.parquet \
  --price  10            # optional GPU-hour price -> enables the $ metric
```

Sample output (all metrics per million tokens, lower is better):

```text
----------  Efficiency summary  ----------
   energy_efficiency (Wh/Mtoken): 84.5
 carbon_efficiency (gCO2/Mtoken): 39.0
 financial_efficiency ($/Mtoken): 2.36
                    total_tokens: 1234567
------------------------------------------
```

For time-and-grid-aware emissions instead, see [kavier carbon](co2.md).

## 11. Running the test-suite & linters {#tests}

CI runs exactly these on every push/PR (tests on Python 3.13 and 3.14), so run them locally first:

```bash
uv run pytest                 # full test suite
uv run ruff check .           # lint
uv run ruff format --check .  # formatting (CI pins ruff==0.15.15; fix with: uv run ruff format .)

# Strict typing is gated INCREMENTALLY — the full tree is not strict-clean yet:
uv run mypy --strict -p kavier.cli -p kavier.ui -p kavier.sdk.co2
uv run mypy --strict --follow-imports=skip \
  src/kavier/__init__.py src/kavier/__main__.py \
  src/kavier/sdk/training/calibration/__init__.py \
  src/kavier/sdk/training/core/engine.py
```

## 12. Troubleshooting {#troubleshooting}

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError` for `kavier.*` | Run `uv sync`, then use `uv run kavier …` (or activate `.venv`). |
| Interactive UI crashes on Windows | The UI is POSIX-only. Use the one-shot CLI subcommands instead. |
| A calibration `UserWarning` about "no entry for…" | Expected — that model/GPU is uncalibrated and falls back to neutral raw physics. See [Training calibration](performance.md#training). |

- Did you find a bug? Please open an issue!
- Did you do any error that others might do? Let us know and we'll add it to this list! See the
  [contributing guidelines](contributing.md).

## 13. Next steps {#next}

1. **Plug in your own traces** (CSV / Parquet with token counts) for inference.
2. **Tune cache policies** – try `--prefix_cache_policy full` vs `prefill`.
3. **Sweep training configs** with `kavier training --input_csv` to compare GPUs, methods and batch
   sizes.
4. **Read the [Architecture](architecture.md)** to see how the pieces connect, then dive into a
   [component page](performance.md).
