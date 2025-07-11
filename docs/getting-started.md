# Getting started

In this mini-tutorial, we will show how to get started with Kavier in under 10 minutes.

## 1. Pre-requisites
1. Python >= 3.9 (**3.11 preferrably**)
2. pip >= 23 & virtualenv / venv

> **Tip for Apple Silicon:** `brew install python@3.11`

## 2. Clone & create a virtual environment
```bash
git clone https://github.com/<you>/kavier.git
cd kavier

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate        
python -m pip install -U pip
```

## 3. Install Kavier
```bash
pip install -e ".[dev]"
```
This 
- Will install main dependencies and tools as defined in pyproject.toml.
- Will resolve runtime deps (e.g., pandas, pyarrow, etc.) and install them.
- Install CLI entry-points
  - `kavier-perf` - for performance simulation
  - `kavier-eff` - for efficiency simulation
- Registers editable packages so changes in `src/` are picked up instantly.

## 4. Run your first simulation

### 4.1 The lonely trace
A tiny trace lives lonely at `src/data/input/input_example.csv` (2436 requests, 100 sessions). 
Visit the LLM Trace Archive and bring some it some fellow traces!

Or... you can add your own traces! Just make sure to follow the format:
- [REQUIRED] `num_input_tokens` (the number of input tokens in the request)
- [REQUIRED] `num_output_tokens` (the number of output tokens in the response)
- [OPTIONAL] `input_tokens` (the actual input tokens, e.g., "[123, 22, 816, 312...]")
- [OPTIONAL] `output_tokens` (the actual output tokens, e.g., "[123, 22, 816, 312...]")
- [OPTIONAL] `session_id` (the session ID, e.g., "1")

### 4.2 Your first simulation
```bash
kavier-perf --trace src/data/input/input_example.csv
```

### 4.3 Your first output
Artifacts will be saved in `src/data/output/`. Here you'll find three files:
- `tasks.parquet`: one row per request,
- `fragments.parquet`: snapshots, each task is broken down into fragments,
- `_sim_results.txt`: a more readable summary.


Congrats! You have just run your first simulation with Kavier! 🎉

## 5. A typical run
```bash
kavier-perf \
  --llm         Llama-3-8B \
  --gpu         A10        \
  --trace       src/data/input/input_example.csv \
  --kv_cache    on \
  --export_rate 0.1 \
  --flush_size  1000
```

See `docs/cli-performance.md` for all flags.

## 6. OpenDC Sustainability
Kavier is integrated with the peer-reviewed [OpenDC](https://opendc.org/) to provide sustainability metrics.  To run a simulation with OpenDC, you need to setup OpenDC, as described in the [OpenDC documentation](https://opendc.org/).

Once set up, use the output of Kavier as the input for OpenDC. This is directly compatible and you don't need to change it! From OpenDC's predictions,
you'll leverage `powerSource.parquet`.


## 7. Post-processing: efficiency metrics

If you ran all the steps so far, including step `6`, you should now have:
- `tasks.parquet` - outputted by Kavier
- `powerSource.parquet` - outputted by OpenDC.

```bash
kavier-eff \
  --kavier data/output_traces/<TIMESTAMP>/tasks.parquet \
  --opendc /path/to/powerSource.parquet \
  --price 10            # GPU-hour price in your currency
```

Sample output:
```bash
----------  Efficiency summary  ----------
 financial_efficiency (EUR / million token / s):       2.36
 sustainability_efficiency (Wh / million token / s):  84.5
 sustainability_efficiency (kgCO2 / million token / s): 0.039
 total_tokens:                                   1 234 567
 total_latency_s:                                  432.10
------------------------------------------
```

## 8. Running the test-suite & linters
```bash
pytest -q          # 100 % pass → confidence in the math
ruff check .       # style & import order
mypy .             # optional static typing

```

## 9. Troubleshooting
| Symptom | Fix |
| ------- | --- |
| `ModuleNotFoundError` for `kavier.*` | Make sure the **venv is active** and you ran:<br>`pip install -e ".[dev]"` |
| *Mypy* says a file is “found twice” | Import modules with the **`kavier.`** prefix, not a relative `` path. |

- Did you find a bug? Please open an issue!
- Did you do any error that others might do? Let us know and we'll add it to this list! See [contributing guidelines](contributing.md).

---

## 10. Next steps

1. **Plug in your own traces** (CSV / Parquet with token counts).
2. **Tune cache policies** – `--prefix_cache_policy full` is especially fun 😉. 
3. **Compare GPUs** by sweeping `--gpu` and plotting metrics from `tasks.parquet`.
4. **Dive into the code** (`src/simulator/`) and adapt the model to your research ideas.

> Happy simulating! 🦙⚡️💻
