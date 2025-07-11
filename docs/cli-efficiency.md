## CLI Kavier-Efficiency

This document provides documentation on how to interact with Kavier Efficiency via the command-line interface (CLI).

| Flag / Option | Type&nbsp;&nbsp;&nbsp; | Default | Description |
|---------------|-----------------|---------|-------------|
| `--kavier` | `path` | **required** | Path to **`tasks.parquet`** produced by `kavier-perf`. |
| `--opendc` | `path` | **required** | Path to **`powerSource.parquet`** produced by the corresponding OpenDC run. |
| `--price` | `float` | `10.0` | GPU-hour price in your local currency, used when computing *financial efficiency*. |
| `--out` | `FILE.json` | _(unset)_ | If set, writes the efficiency summary to the given JSON file in addition to printing it to stdout. |


#### Example: smallest

```bash
kavier-eff \
  --kavier results/run-01/tasks.parquet \
  --opendc  opendc/run-01/powerSource.parquet
```

#### Example: full
```bash
kavier-eff \
  --kavier results/run-01/tasks.parquet \
  --opendc opendc/run-01/powerSource.parquet \
  --price  12.0 \
  --out    results/run-01/efficiency_summary.json
```
