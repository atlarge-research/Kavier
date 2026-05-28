# Kavier Core Formulas

## Training Simulation

### 1. Forward Pass Time
**Formula:** `T_forward = (2 × params × tokens) / (GPU_FLOPS × efficiency) + overhead`

**Source:** [Kaplan et al. (2020) - Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- 2 FLOPs per parameter per token (1 multiply + 1 add)
- Efficiency accounts for memory bandwidth bottlenecks

### 2. Backward Pass Time
**Formula:** `T_backward = 2 × T_forward`

**Source:** Standard deep learning practice ([Goodfellow et al., 2016](http://www.deeplearningbook.org/))
- Backward pass computes gradients (2× forward compute)

### 3. Optimizer Step Time
**Formula:** `T_optimizer = (bytes_to_update / memory_bandwidth) × overhead_factor`

**Source:** Memory-bandwidth limited operation ([Kingma & Ba, 2014 - Adam Optimizer](https://arxiv.org/abs/1412.6980))
- Adam: 4 bytes/param (fp32) or 2 bytes/param (fp16)
- Overhead from kernel launches, synchronization

### 4. Communication Time (Ring All-Reduce)
**Formula:** `T_comm = 2 × (N-1)/N × gradient_bytes / bandwidth`

**Source:** [Thakur et al. (2005) - Optimization of Collective Communication](https://doi.org/10.1177/1094342005051521); [Sergeev & Del Balso (2018) - Horovod](https://arxiv.org/abs/1802.05799); [Li et al. (2020) - PyTorch Distributed](https://arxiv.org/abs/2006.15704)
- N = number of GPUs
- 2× factor: reduce-scatter + all-gather phases
- (N-1)/N: efficiency of ring topology

### 5. Total Step Time
**Formula:** `T_step = T_forward + T_backward + T_optimizer + T_comm`

**Throughput:** `tokens/sec = (batch_size × seq_length × num_GPUs) / T_step`

## Inference Simulation

### 6. Prefill Time
**Formula:** `T_prefill = (2 × params × input_tokens) / (GPU_FLOPS × efficiency)`

**Source:** Same as forward pass, single-pass inference ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))

### 7. Decode Time (per token)
**Formula:** `T_decode = (2 × params) / (GPU_FLOPS × efficiency) + KV_cache_overhead`

**Source:** Autoregressive generation, memory-bound ([Pope et al., 2022 - Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102); [Kwon et al., 2023 - PagedAttention](https://arxiv.org/abs/2309.06180))
- Each token attends to all previous tokens
- KV cache reduces compute but adds memory traffic

### 8. KV Cache Memory
**Formula:** `KV_bytes = 2 × layers × heads × head_dim × seq_length × precision`

**Source:** Transformer architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))
- 2× for Key and Value
- Precision: 2 bytes (fp16) or 4 bytes (fp32)

## GPU Utilization

### 9. Roofline Model
**Formula:** `Performance = min(peak_FLOPS, bandwidth × arithmetic_intensity)`

**Source:** [Williams et al. (2009) - Roofline: An Insightful Visual Performance Model](https://doi.org/10.1145/1498765.1498785)
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

## Sparse / active-parameter models

Kavier does not track expert structure (no `num_experts`, no routing
overhead term). Sparse architectures are encoded by setting the LLMSpec's
`active_params` to the per-token active subset (e.g. Mixtral-8x7B:
`m_params=47B`, `active_params=13B`). The forward-FLOP formula (#1) then
uses `active_params`, so compute scales with the effective per-token cost
rather than the total weight count. Any residual error is absorbed by the
empirical `model_scale` calibration.

## Fine-Tuning Optimizations

### 14. LoRA (Low-Rank Adaptation)
**Formula:** `trainable_params = 2 × rank × (input_dim + output_dim) × num_layers`

**Source:** [Hu et al. (2021) - LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

**Memory Savings:**
- Full fine-tuning: All parameters trainable
- LoRA: Only low-rank matrices trainable (typically <1% of parameters)
- Rank r typically 8-64 for good performance

**Training Time Impact:**
- Forward pass: Same as full model
- Backward pass: Only compute gradients for LoRA weights
- Optimizer: Dramatically reduced memory and time

## Critical Insights

1. **Memory Bottleneck**: Training is memory-bandwidth limited, not compute-limited ([Ivanov et al., 2021](https://arxiv.org/abs/2110.11501))
2. **Batch Size Impact**: Small batches severely underutilize GPU (27× error at batch=1)
3. **Communication Overhead**: Scales with model size and GPU count ([Li et al., 2020](https://arxiv.org/abs/2006.15704))
4. **Efficiency Calibration**: Must be empirically validated per GPU architecture
5. **Sparse models**: encoded via `active_params` only — Kavier does not model expert routing or load balancing.
6. **LoRA Benefits**: Enables fine-tuning of large models with minimal memory overhead

## References

### Foundational Papers
- [Vaswani et al. (2017) - Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [Kaplan et al. (2020) - Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - Compute scaling laws
- [Goodfellow et al. (2016) - Deep Learning Book](http://www.deeplearningbook.org/) - Deep learning fundamentals

### Performance Modeling
- [Williams et al. (2009) - Roofline: An Insightful Visual Performance Model](https://doi.org/10.1145/1498765.1498785) - Performance analysis framework

### Distributed Training
- [Thakur et al. (2005) - Optimization of Collective Communication Operations in MPICH](https://doi.org/10.1177/1094342005051521) - MPI collective operations
- [Sergeev & Del Balso (2018) - Horovod: Fast and Easy Distributed Deep Learning in TensorFlow](https://arxiv.org/abs/1802.05799) - Distributed training framework
- [Li et al. (2020) - PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704) - PyTorch distributed implementation
- [Ivanov et al. (2021) - Data Movement Is All You Need: A Case Study on Optimizing Transformers](https://arxiv.org/abs/2110.11501) - Memory bandwidth analysis

### Optimization Algorithms
- [Kingma & Ba (2014) - Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) - Adam optimizer

### Inference Optimization
- [Pope et al. (2022) - Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) - Inference optimization techniques
- [Kwon et al. (2023) - Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - KV cache optimization

### Sparse-architecture models (supported via `active_params`)
- [Chowdhery et al. (2022) - PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) - Large-scale MoE training
- [Jiang et al. (2024) - Mixtral of Experts](https://arxiv.org/abs/2401.04088) - Mixtral architecture and sparse MoE

### Parameter-Efficient Fine-Tuning
- [Hu et al. (2021) - LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - LoRA fine-tuning method