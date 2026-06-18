# Large inference example (local-only)

`input_example_large.csv` (~74 MB) is the original kavier-inference demo trace —
Llama-tokenized chat token-IDs derived from public chat datasets (LMSYS-Chat-1M /
WildChat, per the PyPI release audit). It is **not shipped in the wheel and not
committed** — its size bloats the package and its dataset license/provenance is
unresolved under kavier's MIT.

The packaged demo trace is instead the tiny synthetic
`src/kavier_inference/data/input/input_example.csv` (~562 B, counts-only).

## Retrieving the large trace

It is preserved in git history. The packaging commit `26a5aac` removed it, so its
parent still has the file:

```
git show 26a5aac~1:src/kavier_inference/data/input/input_example.csv > examples-local/input_example_large.csv
```

Confirm the exact source dataset and its license before any redistribution.
