# Known weaknesses

Honest limitations of Kavier's predictions — places where the model is approximate, uncalibrated, or
unverified. Read these before trusting a number in a high-stakes decision.

## MoE inference is not modeled

Inference uses total parameters, ignoring expert sparsity, so MoE models (Mixtral, Granite-MoE)
over-predict inference latency and energy by roughly 3-4x. Training, by contrast, correctly uses
active parameters.

## Most models run uncalibrated

Only the calibration fit-set carries fitted correction factors; every other model — including the
default Llama-3-8B and both MoE models — falls back to neutral raw physics, unvalidated against
measurements. You can fit your own table for other models with `kavier calibrate <profiling.csv>`
(the any-GPU-count from-scratch fit) and load it via `KAVIER_CALIBRATION=<path>` or
`use_calibration("<path>")`, but until you do, those models stay uncalibrated. The catalog also
carries four **NEEDS-VERIFICATION** model specs (gated or approximated Hugging Face configs) whose
dimensions are unconfirmed, so any calibration or prediction for them inherits that uncertainty.

## The above-8-GPU multi-GPU correction is coupled to the catalog

The shipped calibrations' large-count `multi_gpu_correction` (16/32/64/128 GPUs) is a single global
**median** fit over *every* catalog model's big-GPU runs, so it shifts as the catalog grows. The
recommender only uses predictions at 8 or fewer GPUs, and the frozen thesis files are pinned
(byte-identity guard) only on their 8-or-fewer (used) values — the above-8 entries are allowed to
move. Two related findings: (P1) fitting with above-8-GPU rows in the main joint fit — the uncapped
path `kavier calibrate` takes — makes `comm_scale` data-identifiable (~1.0), whereas the
8-or-fewer-only fit leaves it regularization-pinned (~1.23); and accuracy degrades badly at high GPU
counts (128-GPU held-out MdAPE exceeds ~30%). Treat any above-8-GPU training prediction as out of
the validated regime.

## Calibration may overfit

Throughput rests on seven multiplicative calibration factors, including a flexible
per-(model x method x GPU x count) interaction term. On a small fit-set these can memorise rather
than generalise.

## Headline accuracy isn't reproducible from the package

The held-out validation set behind the ~6.2% MdAPE figure is internal and unshipped, so the two
accuracy tests skip on a clean install — users cannot independently reproduce the published accuracy
claim.

## Calibration is hardware- and dataset-specific

The factors were fit on one GPU cluster and dataset. Predictions for different hardware,
interconnects, or data distributions extrapolate beyond the fitted regime and may degrade.

## Two divergent energy methodologies

Energy can come from the self-contained GPU-power estimate (API/UI) or from an external OpenDC power
simulation (more precise). The two paths can return different energy for the same workload.

## Hardcoded engine constants

The training engine hardcodes rules of thumb — optimizer ~20 bytes/parameter, ~5 memory passes per
step, backward = 2x forward — applied uniformly across models and hardware rather than derived per
case.

## Utilization and throughput use different bases

Reported GPU compute utilization is the raw MFU, while throughput applies the calibrated scale — so
the displayed utilization is not consistent with the displayed throughput for the same run.
