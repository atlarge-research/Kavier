"""
Communication simulation for distributed training.

Simulates gradient synchronization across GPUs using hierarchical all-reduce:
intra-node via NVLink, inter-node via InfiniBand.

References:
- Culler et al. 1993: "LogP: Towards a Realistic Model of Parallel Computation"
- Thakur et al. 2005: "Optimization of Collective Communication Operations"
"""

from __future__ import annotations

import math

INFINIBAND_GBPS = 200.0


def _ring_allreduce_time(
    gradient_bytes: float,
    num_participants: int,
    bandwidth_gbps: float,
    latency_s: float = 5e-6,
    overhead_per_msg_s: float = 2e-6,
) -> float:
    """Ring all-reduce time for *num_participants* using bandwidth *bandwidth_gbps*."""
    if num_participants <= 1:
        return 0.0
    bw_bytes_per_sec = bandwidth_gbps * 1e9 / 8
    chunk = gradient_bytes * (num_participants - 1) / num_participants
    return (
        latency_s * math.log2(num_participants)
        + overhead_per_msg_s * (num_participants - 1)
        + chunk / bw_bytes_per_sec
    )


def simulate_allreduce(
    trainable_params: int,
    num_gpus: int,
    network_bandwidth_gbps: float,
    num_nodes: int = 1,
) -> float:
    """
    Simulate hierarchical all-reduce for distributed training.

    When training spans multiple nodes the gradient sync is split into:
      1. Intra-node reduce-scatter  (NVLink / PCIe — *network_bandwidth_gbps*)
      2. Inter-node all-reduce      (InfiniBand — INFINIBAND_GBPS)
      3. Intra-node all-gather      (same as step 1, folded into the estimate)

    For single-node training the classic ring all-reduce is used.

    Args:
        trainable_params: Number of trainable parameters.
        num_gpus: **Total** number of GPUs across all nodes.
        network_bandwidth_gbps: Intra-node GPU bandwidth (NVLink / PCIe).
        num_nodes: Number of physical nodes.

    Returns:
        Communication time in seconds.
    """
    if num_gpus <= 1:
        return 0.0

    gradient_bytes = trainable_params * 4  # FP32

    from kavier_training.core.calibration import get_comm_scale

    if num_nodes <= 1:
        return _ring_allreduce_time(gradient_bytes, num_gpus, network_bandwidth_gbps) * get_comm_scale()

    gpus_per_node = max(1, num_gpus // num_nodes)

    intra = _ring_allreduce_time(gradient_bytes, gpus_per_node, network_bandwidth_gbps)
    inter = _ring_allreduce_time(gradient_bytes, num_nodes, INFINIBAND_GBPS)

    return (intra + inter) * get_comm_scale()

