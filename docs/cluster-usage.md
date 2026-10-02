# Cluster simulator

`kavier.sdk.cluster` simulates a fixed-size GPU cluster that runs jobs of known duration under
one of four scheduling policies: `distributed-fcfs`, `distributed-backfill`, `consolidated-fcfs`
(default), or `consolidated-backfill`.

Per job, it reports queue wait, start and end time, runtime, energy, and `goodput`. Per cluster, it
reports makespan, utilization, peak GPUs in use, and peak queue depth, plus a timeline of GPUs in use
and queue depth.

There are two cluster goodput measures. `goodput_jobs_per_s` is the scheduling throughput in jobs/s.
`scheduling_goodput` is the scheduling efficiency, `sum(runtime_s) / sum(turnaround_s)`: training time
over wall-clock time.

Python:

```python
from kavier.sdk.cluster import schedule
r = schedule([{"submit_s": 0, "gpus": 8, "duration_s": 3600}],
             policy="consolidated-fcfs", num_nodes=2, node_gpus=8)
print(r.cluster.makespan_h, r.cluster.utilization)
```

CLI:

```bash
kavier cluster --jobs jobs.csv --policy consolidated-backfill --num-nodes 2 --node-gpus 8
```

Jobs CSV: `submit_s,gpus,duration_s[,nodes,power_w_per_gpu]`.
