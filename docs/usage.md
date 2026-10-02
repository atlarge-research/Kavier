# Usage

## CLI

Every subcommand documents its own flags.

```bash
kavier --help           # inference, training, cluster, energy, carbon
kavier training --help
```

An example inference trace ships with the package:

```bash
TRACE=$(python -c "from importlib.resources import files; print(files('kavier.sdk.inference') / 'data/input/input_example.csv')")
kavier inference --trace "$TRACE"
```

## Python API

A runnable script: `python docs/usage.py`.

```python
--8<-- "docs/usage.py"
```
