# SXM4 Accuracy Investigation - Summary

## Problem
NVIDIA A100-SXM4-80GB had 72-81% prediction error, while A100-PCIe and H100-PCIe achieved ~10% error.

## Root Causes Identified

### 1. Hardcoded Network Bandwidth
**Issue**: `simulate_allreduce()` used hardcoded 400 Gbps for all GPUs
- SXM4 has NVLink 3.0: ~600 GB/s = 4800 Gbps (12x faster)
- PCIe has PCIe 4.0 x16: ~8 GB/s = 64 Gbps

**Fix**: Added `network_bandwidth_gbps` to GPUSpec and pass GPU-specific values

### 2. Multi-GPU Correction Factors
**Issue**: Band-aid corrections (2.6-3.5x) masked underlying physics model issues
- These were empirical patches hiding the real problems

**Fix**: Removed all hardcoded correction factors, relying on physics-based model

### 3. Data Quality Issues
**Issue**: Validation data contained duplicate configurations with wildly different measurements
- Same config (model, batch, GPUs) had 2.5x throughput variance
- Caused by mixing 1-node vs 2-node runs with same GPU count

**Example**:
```
llama3.1-8b, batch=8, 8 GPUs, 8192 tokens:
- 2 nodes: 8,217 tokens/s
- 1 node: 20,755 tokens/s  (2.5x faster!)
```

**Fix**: Created `validation_clean.csv` filtering for single-node runs only
- Original: 3,882 samples
- After filtering: 3,350 samples (1-node only)
- After deduplication: 2,425 samples

## Current Status

### After Fixes
Even with clean data and proper NVLink bandwidth, calibration still hits MFU=1.0 upper bound with 44-62% error. This indicates the base physics model is still missing something fundamental.

### Possible Remaining Issues

1. **Communication Model**: Ring all-reduce formula may not accurately model NVLink topology
2. **Batch Size Scaling**: MFU may need to scale differently with batch size for multi-GPU
3. **Memory Contention**: Multi-GPU memory bandwidth contention not modeled
4. **Synchronization Overhead**: Barrier synchronization time not accounted for

## Next Steps

### Option A: Add Small Empirical Corrections
- Keep physics-based model
- Add small (<1.2x) corrections for synchronization overhead
- Calibrate per GPU count (1, 2, 4, 8 GPUs)

### Option B: Improve Communication Model
- Research actual NVLink topology (ring vs tree vs all-to-all)
- Model memory bandwidth contention in multi-GPU scenarios
- Add synchronization barrier time

### Option C: Separate Single-GPU vs Multi-GPU Models
- Use current model for single-GPU (works well for PCIe)
- Develop separate multi-GPU model with proper scaling factors
- Validate each independently

## Recommendation
Start with **Option A** for quick improvement, then pursue **Option B** for scientific rigor. The current 44-62% error with clean data suggests we're close but missing a ~1.5-2x scaling factor for multi-GPU scenarios.

## Files Modified
- `src/library/specs/GPUSpec.py` - Added network_bandwidth_gbps parameter
- `src/library/gpu.py` - Updated GPU specs with correct bandwidths
- `src/kavier_training/components/communication.py` - Made bandwidth GPU-specific
- `src/kavier_training/core/engine.py` - Removed correction factors, pass GPU bandwidth
- `src/kavier_training/validation/clean_validation_data.py` - Data cleaning script
- `src/kavier_training/validation/calibrate_mfu.py` - MFU calibration script
- `src/kavier_training/validation/debug_predictions.py` - Debugging script

## Key Learnings
1. Always validate data quality before blaming the model
2. Empirical corrections mask root causes - fix physics first
3. Multi-node vs single-node performance can differ dramatically
4. Communication bandwidth is critical for multi-GPU accuracy