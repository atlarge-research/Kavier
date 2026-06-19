"""Infrastructure defaults for the training engine (cluster/network constants)."""

# Inter-node interconnect bandwidth (Gbps) for the multi-node all-reduce.
# 200 = HDR InfiniBand; override per cluster (100 = EDR, 400 = NDR).
INFINIBAND_GBPS = 200.0
