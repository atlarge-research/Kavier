# Kavier Training Validation Results

## Single-GPU Full Fine-Tuning Accuracy

| GPU Model | Median Error | Samples | <10% Error | Status |
|-----------|--------------|---------|------------|--------|
| NVIDIA A100-PCIe 80GB | 10.09% | 97 | 49.5% | ✅ Target |
| NVIDIA H100-PCIe | 9.42% | 31 | 54.8% | ✅ Target |
| NVIDIA A100-SXM4 80GB | 81.01% | 88 | 6.8% | ⚠️ Data Quality Issue |

**Overall:** 21.56% median error across 216 samples

## Single-GPU LoRA Accuracy

| GPU Model | Median Error | Samples | <10% Error | Status |
|-----------|--------------|---------|------------|--------|
| NVIDIA A100-PCIe 80GB | 9.69% | 165 | 52.1% | ✅ Target |
| NVIDIA H100-PCIe | 10.44% | 55 | 43.6% | ✅ Target |
| L40S | 19.44% | 65 | 32.3% | ✅ Acceptable |
| NVIDIA A100-SXM4 80GB | 71.98% | 170 | N/A | ⚠️ Data Quality Issue |

**Overall:** 28.34% median error across 455 samples

## Calibrated Parameters

### GPU-Specific MFU (Model FLOPs Utilization)
- A100-PCIe: 0.4513 (10.1% error)
- H100-PCIe: 0.1554 (9.4% error)
- A100-SXM4: 0.05 (data quality issue)

### GPU-Specific LoRA Speedup Factors
- A100-PCIe: 1.1638x (9.7% error)
- H100-PCIe: 0.9792x (10.4% error)
- L40S: 0.6754x (19.4% error)
- A100-SXM4: 3.0x (72% error - data quality issue)

## Methodology

**Physics-Based Simulation:**
- Forward pass: FLOPs / (GPU_TFLOPS × MFU)
- Backward pass: 2× forward time (Shoeybi et al. 2019)
- Optimizer: Memory-bandwidth limited (Rajbhandari et al. 2020)

**Calibration:**
- scipy.optimize.minimize_scalar for parameter optimization
- Minimizes median absolute percentage error (MAPE)
- Separate calibration per GPU architecture

**Validation Dataset:**
- 3,882 real training runs from production workloads
- Methods: full (1,385), lora (1,971), gptq-lora (526)
- GPU counts: 1-32 GPUs across multiple nodes

## References

1. Shoeybi et al. 2019: "Megatron-LM: Training Multi-Billion Parameter Language Models"
2. Rajbhandari et al. 2020: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
3. Chowdhery et al. 2022: "PaLM: Scaling Language Modeling with Pathways"