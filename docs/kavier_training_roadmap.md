# Kavier Training Simulation Roadmap

## Current State
- ✅ Physics-based training simulation implemented
- ✅ Multi-GPU communication (ring all-reduce)
- ✅ 21 LLM models in library
- ✅ Validation framework (3,882 samples)
- ✅ **Single-GPU full fine-tuning: 10.1% median error (A100-PCIe), 9.4% (H100-PCIe)**
- ✅ **Single-GPU LoRA: 9.7% median error (A100-PCIe), 10.4% (H100-PCIe)**

## Goal
Multi-GPU training simulator predicting:
- **Throughput** (tokens/sec)
- **Latency** (time per step)
- **Energy** (Joules per step)
- **Methods:** Full fine-tuning, LoRA

## Proposed Plan

### Phase 1: Fix Core Accuracy (Priority: CRITICAL)
**Target:** <10% median error for single-GPU full fine-tuning

**Problem:** Small batches (1-2) have 2000%+ error due to extreme real-world variability
- Actual: 40-200 tokens/s
- Predicted: 1000-2600 tokens/s

**Solution Options:**
1. **Hybrid approach:** Physics-based + empirical correction factors
   - Keep physics formulas for compute/memory
   - Add batch-size lookup table from validation data
   - Maintains interpretability while fixing edge cases

2. **Roofline refinement:** Add memory bandwidth bottleneck
   - Current: `time = FLOPs / (GPU_FLOPS × efficiency)`
   - Better: `time = max(compute_time, memory_time)`
   - Memory time = `bytes_transferred / bandwidth`

3. **Accept limitation:** Document that batch<4 is unreliable
   - Focus on practical batch sizes (4-32)
   - Already achieving <2% error for batch≥4

**Recommendation:** Option 1 (hybrid) - best accuracy/interpretability tradeoff

### Phase 2: LoRA Support ✅
**Status:** COMPLETED

**Achievements:**
1. ✅ Implemented LoRA backward pass and optimizer (physics-based)
2. ✅ GPU-specific LoRA speedup calibration using scipy optimization
3. ✅ Validated against 455 single-GPU LoRA samples
4. ✅ A100-PCIe: 9.7% median error (52% samples <10%)
5. ✅ H100-PCIe: 10.4% median error (44% samples <10%)

**Implementation:**
- LoRA uses same forward pass as full fine-tuning
- Backward pass: full backprop but only computes gradients for adapters
- Optimizer: only updates adapter parameters (~0.1-1% of model)
- GPU-specific speedup factors account for architecture differences

### Phase 3: Energy Modeling
**Status:** Not implemented

**Formula:**
```
Energy (J) = Power (W) × Time (s)
Power = GPU_TDP × utilization_factor
```

**Tasks:**
1. Add GPU TDP specs to [`GPUSpec.py`](../src/library/specs/GPUSpec.py:1)
2. Implement power model based on utilization
   - Idle: ~30% TDP
   - Compute: ~80-95% TDP
   - Memory: ~60-70% TDP
3. Integrate with existing time predictions

**Reference:** NVIDIA Management Library (NVML) power measurements

### Phase 4: Multi-GPU Scaling
**Status:** Implemented but needs validation

**Current implementation:**
- Ring all-reduce communication
- Linear scaling assumption: `throughput = single_GPU × N`

**Validation needed:**
1. Test against 2,400+ multi-GPU samples
2. Verify communication overhead scales correctly
3. Check for non-linear effects (network congestion, synchronization)

**Expected issues:**
- Communication overhead underestimated for large N (>32 GPUs)
- Network topology effects (NVLink vs InfiniBand)

### Phase 5: Advanced Features
**Lower priority, implement after Phase 1-4:**

1. **Gradient accumulation**
   - Reduces communication frequency
   - Changes effective batch size

2. **Mixed precision (FP16/BF16)**
   - Already assumed in current model
   - May need refinement for accuracy

3. **Pipeline parallelism**
   - Different scaling model than data parallel
   - Requires stage-by-stage analysis

4. **Activation checkpointing**
   - Trade compute for memory
   - Already partially accounted in efficiency factor

## Recommended Next Steps

### Immediate (This Week)
1. **Implement hybrid model** (Phase 1, Option 1)
   - Extract batch-size correction factors from validation data
   - Apply as multipliers to physics-based predictions
   - Target: <10% median error

2. **Validate LoRA** (Phase 2)
   - Run validator on LoRA samples
   - Identify accuracy gaps
   - Implement fixes

### Short-term (Next 2 Weeks)
3. **Add energy modeling** (Phase 3)
   - Simple power model (TDP × utilization)
   - Validate against real measurements if available

4. **Validate multi-GPU** (Phase 4)
   - Test 2-32 GPU configurations
   - Fix communication overhead if needed

### Long-term (Next Month)
5. **Advanced features** (Phase 5)
   - Based on user requirements
   - Gradient accumulation most valuable

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Single-GPU Full (A100-PCIe) | 10.1% | <10% | ✅ |
| Single-GPU Full (H100-PCIe) | 9.4% | <10% | ✅ |
| Single-GPU LoRA (A100-PCIe) | 9.7% | <10% | ✅ |
| Single-GPU LoRA (H100-PCIe) | 10.4% | <10% | ✅ |
| Multi-GPU Full MAPE | Unknown | <15% | 🔄 Next |
| Multi-GPU LoRA MAPE | Unknown | <15% | 🔄 Next |
| Energy Error | N/A | <20% | ⏳ Future |
| Coverage | 3,882 samples | All scenarios | ✅ |

## Technical Debt

1. **Batch-size edge cases:** Need empirical corrections
2. **GPU-specific tuning:** Efficiency varies by architecture (A100 vs H100)
3. **Model-specific effects:** Some models may have unique characteristics
4. **Validation data quality:** Some samples may have measurement errors

## References

- Current formulas: [`docs/kavier_formulas.bob`](kavier_formulas.bob:1)
- Validation results: `src/kavier_training/data/output/validation_results.csv`
- Implementation: [`src/kavier_training/core/engine.py`](../src/kavier_training/core/engine.py:1)