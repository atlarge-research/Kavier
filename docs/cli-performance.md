## CLI Kavier-Performance

This document provides documentation on how to interact with Kavier Performance via the command-line interface (CLI).



| Flag / Option | Type&nbsp;&nbsp;&nbsp; | Default | Description |
|---------------|-----------------|---------|-------------|
| `--llm` | `str` | `Llama-3-8B` | Key of the LLM to simulate (see `LLM_SPEC_LIBRARY`). |
| `--gpu` | `str` | `A10` | GPU model (see `GPU_SPEC_LIBRARY`). |
| `--trace` | `path` | `../data/input/input_example.csv` | Input trace containing per-request token counts (CSV or Parquet). |
| `--output_folder` | `path` | `../data/output_traces` | Destination directory for generated Parquet files & summary. |
| `--kv_cache` | `on \| off` | `on` | Enable/disable vLLM-style KV caching. |
| `--prefix_len` | `int` | `1024` | Minimum prompt length to enter the **prefix cache** (0 ⇒ disable). |
| `--export_rate` | `float` | `0.1` | Snapshot interval **in seconds** for utilisation traces. |
| `--flush_size` | `int` | `1000` | Rows to buffer before writing Parquet (<code>0 → one-shot</code>). |
| `--prefix_cache_min_tokens` | `int` | `1024` | Same as `--prefix_len`, kept for backwards-compatibility. |
| `--max_cached_prompts` | `int` | `10` | Capacity of the prefix cache (LRU). |
| `--cache_scope` | `session \| global` | `session` | Whether the cache key includes `session_id`. |
| `--prefix_cache_policy` | `none \| prefill \| full` | `prefill` | *prefill*: skip prefill on hit. *full*: skip prefill **and** decode. |

#### Example: smallest

```bash
kavier-perf
```


#### Example: medium
```bash
kaiver-perf  \
  --llm         Llama-3-8B \
  --gpu         A10 \
  --trace       ../data/input/your_special_trace.csv \
  --kv_cache    on \
  --prefix_len  1024 \
  --export_rate 0.1 \
  --flush_size  1000
```

#### Example: full
```bash
kavier-perf \
  --llm                     Llama-3-8B       \
  --gpu                     A10              \
  --trace                   traces/workload.csv \
  --output_folder           results/run-01   \
  --kv_cache                on               \
  --prefix_len              1024             \
  --export_rate             0.05             \
  --flush_size              1000             \
  --prefix_cache_min_tokens 1024             \
  --max_cached_prompts      10               \
  --cache_scope             session          \
  --prefix_cache_policy     prefill
```
