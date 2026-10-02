# Fine-tuning model

As an analytical predictor, Kavier's predictions are based on properties of the fine-tuning experiment.
At the core, Kavier first computes the step runtime and then derives the throughput.

## Time per step

A four-term sum: the forward pass, the backward pass, the optimizer update, and the gradient
synchronization across GPUs.

$$T_s = G_a \times (T_f + T_b) + T_o + T_c$$

$G_a$ is the number of gradient-accumulation micro-steps (default $1$).

## Forward pass and MFU

$$T_f = \frac{2 \times P \times B \times S}{F \times E} + O_t
\qquad
E = E_b \times E_g \times \min\bigl(1,\; a_1 \log_2 B + a_2\bigr)$$

$P$ is the number of model parameters used in the forward pass (the active parameters of an MoE model), $B$ the micro-batch size, $S$ the sequence
length in tokens, $F$ the GPU's peak FP16 Tensor Core throughput, $E$ the effective MFU, and $O_t$ the
calibrated training overhead. $E_b$ is the GPU's nominal MFU factor, $E_g$ a per-GPU calibration factor,
and $a_1, a_2$ are calibration factors.

## Backward pass

$$T_b = 2 \times T_f$$

## Optimizer update

AdamW, dominated by memory traffic: 20 bytes moved per trainable parameter, each step.

$$T_o = \frac{20 \times P_t}{B_m}
\qquad
P_t = \begin{cases} P_{all} & \text{full fine-tuning} \\ 2 \times r \times d \times k \times L & \text{LoRA, GPTQ-LoRA, QLoRA, aLoRA} \end{cases}$$

$B_m$ is the GPU memory bandwidth, $P_{all}$ the total number of model parameters, $r = 8$ the LoRA rank,
$d$ the hidden size, $k = 4$ the number of target modules per layer, and $L$ the number of transformer layers.

## Gradient communication

A ring all-reduce: none on one GPU, one ring at intra-node bandwidth on one node, and an intra-node then
inter-node (InfiniBand) all-reduce across nodes.

$$T_c = \begin{cases} 0 & G = 1 \\ c_c \times T_r(G, W_n) & N = 1 \\ c_c \times \bigl(T_r(G_n, W_n) + T_r(N, W_i)\bigr) & N > 1 \end{cases}
\qquad
T_r(p, W) = \ell \log_2 p + o\,(p - 1) + \frac{D\,(p - 1)}{p \times W / 8}$$

$G$ is the total number of GPUs, $N$ the number of nodes, and $G_n = \max(1, \lfloor G / N \rfloor)$ the GPUs per node. $W_n$ is the
GPU interconnect rate and $W_i = 200$ Gbit/s the InfiniBand rate. $D = 4 \times P_t$ is the gradient size in
bytes, $\ell$ the per-hop latency, $o$ the per-message overhead, and $c_c$ a calibrated communication scale.

## Throughput

$$T = \frac{G_a \times B \times S \times G}{m_g \times T_s} \times c_m \times c_o \times c_i$$

$c_m$, $c_o$, and $c_i$ are calibration scales per fine-tuning method, per LLM, and per (model, method,
GPU, $G$) interaction; $m_g$ is a multi-GPU correction.

## Calibration

Pure physics-driven modeling is not enough when simulating real-world large-scale systems, so Kavier is
calibrated in two tiers on `LLMFineTuningBench`:

1. Global: one correction factor per GPU model, fine-tuning method, LLM, GPU count, and one for
   communication, fitted together on the training split (70%) with Powell's method.
2. Four-way: for each (LLM, method, GPU type, GPU count) group, the median ratio of measurement to
   the tier-1 prediction.

On the held-out test split, calibration reduces the MdAPE, averaged over the four dense models, from 12.5% to 5.4%
(MSc thesis, E1).

Calibration is keyed on exact catalog names. An uncalibrated name falls back to a neutral 1.0.
