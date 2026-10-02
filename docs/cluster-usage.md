# Cluster simulator

`kavier.sdk.cluster` simulates a fixed-size GPU cluster running jobs of
**known duration** under a scheduling policy: `distributed-fcfs`, `distributed-backfill`,
`consolidated-fcfs` (default), or `consolidated-backfill`. It reports **per-job** metrics (queue wait,
start/end, runtime, energy, per-job `goodput`) and **per-cluster** metrics (makespan,
utilization, peak GPUs/queue) plus a GPUs-in-use/queue-depth timeline. Two goodput measures are
reported: `goodput_jobs_per_s` (scheduling throughput, jobs/s) and `scheduling_goodput`
(scheduling efficiency = `Σ runtime_s / Σ turnaround_s`, i.e. training time / wall-clock).

**Python**

```python
from kavier.sdk.cluster import schedule
r = schedule([{"submit_s": 0, "gpus": 8, "duration_s": 3600}],
             policy="consolidated-fcfs", num_nodes=2, node_gpus=8)
print(r.cluster.makespan_h, r.cluster.utilization)
```

**CLI**

```bash
kavier cluster --jobs jobs.csv --policy consolidated-backfill --num-nodes 2 --node-gpus 8
```

Jobs CSV: `submit_s,gpus,duration_s[,nodes,power_w_per_gpu]`.
