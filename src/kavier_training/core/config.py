from library.specs.GPUSpec import GPUSpec

BATCH_ALPHA = 0.0341
BATCH_BETA = 0.8147
SEQ_GAMMA = 0.1781
SEQ_DELTA = 3.5714


def get_training_compute_efficiency(batch_size: int, seq_length: int, gpu_spec: GPUSpec) -> float:
    import math

    from kavier_training.core.calibration import get_mfu_multiplier

    base_mfu = gpu_spec.mfu_factor * get_mfu_multiplier(gpu_spec.name)
    batch_scale = min(1.0, BATCH_ALPHA * math.log2(batch_size) + BATCH_BETA)
    seq_scale = min(1.0, SEQ_GAMMA * math.log2(seq_length / 512) + SEQ_DELTA)
    return float(base_mfu * batch_scale * seq_scale)
