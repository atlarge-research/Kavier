e# Kavier Core Formulas

## Training Simulation

### 1. Forward Pass Time
**Formula:** `T_forward = (2 × params × tokens) / (GPU_FLOPS × efficiency) + overhead`

**Source:** Kaplan et al. (2020) - Scaling Laws for Neural Language Models
- 2 FLOPs per parameter per token (1 multiply + 1 add)
- Efficiency accounts for memory bandwidth bottlenecks

### 2. Backward Pass Time
**Formula:** `T_backward = 2 × T_forward`

**Source:** Standard deep learning practice
- Backward pass computes gradients (2× forward compute)

### 3. Optimizer Step Time
**Formula:** `T_optimizer = (bytes_to_update / memory_bandwidth) × overhead_factor`

**Source:** Memory-bandwidth limited operation
- Adam: 4 bytes/param (fp32) or 2 bytes/param (fp16)
- Overhead from kernel launches, synchronization

### 4. Communication Time (Ring All-Reduce)
**Formula:** `T_comm = 2 × (N-1)/N × gradient_bytes / bandwidth`

**Source:** Thakur et al. (2005) - Optimization of Collective Communication
- N = number of GPUs
- 2× factor: reduce-scatter + all-gather phases
- (N-1)/N: efficiency of ring topology

### 5. Total Step Time
**Formula:** `T_step = T_forward + T_backward + T_optimizer + T_comm`

**Throughput:** `tokens/sec = (batch_size × seq_length × num_GPUs) / T_step`

## Inference Simulation

### 6. Prefill Time
**Formula:** `T_prefill = (2 × params × input_tokens) / (GPU_FLOPS × efficiency)`

**Source:** Same as forward pass, single-pass inference

### 7. Decode Time (per token)
**Formula:** `T_decode = (2 × params) / (GPU_FLOPS × efficiency) + KV_cache_overhead`

**Source:** Autoregressive generation, memory-bound
- Each token attends to all previous tokens
- KV cache reduces compute but adds memory traffic

### 8. KV Cache Memory
**Formula:** `KV_bytes = 2 × layers × heads × head_dim × seq_length × precision`

**Source:** Transformer architecture
- 2× for Key and Value
- Precision: 2 bytes (fp16) or 4 bytes (fp32)

## GPU Utilization

### 9. Roofline Model
**Formula:** `Performance = min(peak_FLOPS, bandwidth × arithmetic_intensity)`

**Source:** Williams et al. (2009) - Roofline: An Insightful Visual Performance Model
- Arithmetic intensity = FLOPs / bytes_transferred
- Training typically memory-bound (low AI)

### 10. Compute Efficiency
**Formula:** `efficiency = base_eff × batch_scale_factor × model_scale_factor`

**Factors:**
- Batch size: Larger batches → better parallelism
- Model size: Larger models → better compute/memory ratio
- Typical training: 15-30% of peak FLOPS

## Key Parameters

### Model Parameters
- `params`: Total trainable parameters
- `layers`: Number of transformer layers
- `hidden_dim`: Hidden dimension size
- `heads`: Number of attention heads

### Hardware Parameters
- `GPU_FLOPS`: Peak TFLOPS (fp16/bf16)
- `memory_bandwidth`: GB/s
- `network_bandwidth`: GB/s (NVLink, InfiniBand)

### Training Parameters
- `batch_size`: Samples per GPU per step
- `seq_length`: Tokens per sample
- `num_GPUs`: Total GPUs (data parallel)
- `gradient_accumulation`: Steps before optimizer update

## Critical Insights

1. **Memory Bottleneck**: Training is memory-bandwidth limited, not compute-limited
2. **Batch Size Impact**: Small batches severely underutilize GPU (27× error at batch=1)
3. **Communication Overhead**: Scales with model size and GPU count
4. **Efficiency Calibration**: Must be empirically validated per GPU architecture

## References

- Kaplan et al. (2020): Scaling Laws for Neural Language Models
- Williams et al. (2009): Roofline Model
- Thakur et al. (2005): MPI Collective Communication Optimization
- Vaswani et al. (2017): Attention Is All You Need (Transformer architecture)