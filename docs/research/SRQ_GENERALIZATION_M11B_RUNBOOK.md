# SRQ generalization M11b: same-byte INT8 scale refinement

Status: implementation ready; the real train-only run is pending. M11b does
not authorize test-feature extraction or held-out test evaluation.

## Question

M6 showed that fixed max-absolute P2B INT8 narrowly missed the preregistered
0.25-point AIA-retention gate at width 20,000. M11 recovered that gate by
promoting selected factor blocks to FP16, but paid for a precision mask and a
larger persistent payload. M11b asks a narrower question:

> Can a better scale for the same INT8 codes reduce the error without adding
> any persistent byte relative to P2B?

M11b is a one-shot development follow-up. It neither replaces the disclosed
M6 failure nor invalidates the successful M11 adaptive result.

## Frozen scale rule

The diagonal remains FP32 and every strict-upper entry remains INT8. For each
group, P2B's scale

`max(abs(values)) / 127`

is used as the initializer. Four fixed alternating steps are then applied:

1. assign each value to its nearest clipped integer in `[-127, 127]`;
2. with those integer assignments fixed, compute the least-squares scale
   `(values dot codes) / (codes dot codes)`;
3. reassign the nearest INT8 codes under the new scale;
4. accept the candidate only if its squared reconstruction error does not
   increase.

The rule reads only the current values in one factor group. It does not read
labels, logits, margins, validation accuracy, or test data. The checkpoint
stores exactly the same INT8 value tensors and FP32 scale tensors as P2B;
there is no precision mask, error-feedback residual, or optimizer state.

This is a deterministic fixed-iteration form of codebook-scale refinement.
The general motivation is consistent with least-squares scale optimization in
quantization, but M11b's exact rule and its applicability to streaming
square-root Ridge are evaluated here rather than inherited as a claim from a
different model family.

## Comparison and source lock

M11b is locked to both existing artifacts:

- `srq_generalization_m6_width_sweep_train_only.zip`, SHA-256
  `b2739b9da023ebd2eedb6fdfe01c394e94f252773e847533b35350021c3d239e`;
- `srq_generalization_m11_adaptive_precision_train_only.zip`, SHA-256
  `65ce03df4da7041833014628b59aac1167f77bde2d64348b9a8a1e4fe09370a7`.

It reuses the M6 training cache, class order, train/validation split,
projection prefixes, width-specific Ridge values, and Exact/P2B/FP16
references. Only widths 10,000 and 20,000 are rerun. Exact is rerun as a
reproduction sentinel; the refined all-INT8 method is the only new method.
The M11 adaptive values are reported as a transparent accuracy/state/time
reference, not used to select or modify the M11b rule.

## Gates

- all M6/M11 source-identity checks pass;
- both widths complete;
- Exact AIA and final accuracy reproduce M6 within $10^{-6}$ percentage
  points;
- refined INT8 and P2B have identical persistent tensor bytes after every
  task;
- refined factor reconstruction error is no larger than max-absolute P2B on
  the same pre-quantization factor after every task;
- refined AIA loss relative to Exact is at most 0.25 point at both widths;
- refined AIA is no worse than archived P2B at both widths;
- maximum solver relative residual is at most `1e-5`.

Failure is scientifically valid. The iteration count, grouping, acceptance
rule, Ridge values, or gates must not be changed after seeing the result.

## Run

Open `notebooks/srq_generalization_m11b_scale_refined_colab.ipynb` in a fresh
T4 runtime and execute every cell in order. Upload the exact M6 and M11 ZIPs
when requested. The notebook materializes only `train.pt`, refuses a visible
`test.pt`, and downloads:

`srq_generalization_m11b_scale_refined_train_only.zip`

Return the ZIP whether the formal status is PASS or FAIL.

## Permitted conclusion

A PASS supports the specific claim that fixed least-squares scale refinement
improves or preserves P2B validation AIA at widths 10k and 20k without adding
persistent state. It does not prove optimal quantization, superiority on test
data, stability at arbitrary widths/task frequencies, or replacement of the
adaptive M11 Pareto point.
