# SRQ generalization M11: budget-locked adaptive precision

Status: implementation ready; the real train-only run is pending. M11 does
not authorize test-feature extraction or held-out test evaluation.

## Question

M6 showed that fixed P2B INT8 narrowly misses the predeclared 0.25-point AIA
retention gate at width 20,000, while FP16 remains close to Exact. M11 asks:

> Can a small, fixed FP16 allowance reduce factor error enough to recover the
> 10k/20k retention gate without giving up most of P2B's state reduction?

This is a development-only follow-up to a disclosed failure. It is not a new
test result and cannot be used to hide or replace M6.

## Frozen adaptive rule

The diagonal remains FP32. Every strict-upper block is evaluated under the
same groupwise INT8 encoding as P2B and under FP16 storage. For each block,
the implementation computes

`(INT8 squared error - FP16 squared error) / additional bytes`.

It then promotes blocks in descending score order until reaching 25% of the
byte interval between all-INT8 and all-FP16 strict-upper storage. Ties use the
ascending upper-block index. The rule reads only the current factor values;
it does not read labels, logits, margins, validation accuracy, or test data.
The block precision mask is stored as an explicit uint8 tensor and counted in
persistent state.

The 25% allowance is locked before running M11. No alternative percentage,
selection score, group size, block size, or fallback retry may be tried after
observing the output.

## Comparison and source lock

M11 reuses the exact training cache, class order, train/validation split,
projection prefixes, width-specific Ridge values, and reference results from
`srq_generalization_m6_width_sweep_train_only.zip` (SHA-256
`b2739b9da023ebd2eedb6fdfe01c394e94f252773e847533b35350021c3d239e`).
The archived `m6_results.json` SHA-256 is
`648630c5f0b70ed85e675942a8c2fb10e6f33eb3ff8f03e8a71421de22f44cbb`.

Only widths 10,000 and 20,000 are rerun. Exact is rerun as a reproduction
sentinel, while FP16 and fixed P2B reference values are read from the locked
M6 artifact. The runner refuses execution unless the checkpoint identity,
class set, class order, partition hashes, full projection, and both
projection-prefix hashes match M6. Exact AIA and final accuracy must reproduce
the M6 values within $10^{-6}$ percentage points before the adaptive result is
accepted.

## Gates

- all M6 identity checks pass;
- both widths complete;
- Exact AIA and final accuracy reproduce the source M6 artifact within
  $10^{-6}$ percentage points;
- actual factor bytes, including the uint8 mask and every scale, remain below
  the precomputed ceiling after every task;
- adaptive local factor error is never larger than the internally evaluated
  all-INT8 error for the same pre-quantization factor;
- final adaptive state lies strictly between P2B and FP16 state at each width;
- adaptive AIA loss relative to archived Exact is at most 0.25 point at each
  width;
- adaptive AIA is no worse than archived P2B at each width;
- adaptive total-state reduction relative to Exact is at least 75% at each
  width;
- the maximum solver relative residual is at most `1e-5`.

Failure is scientifically valid. No gate or adaptive policy may be relaxed
after observing M11 output.

## Run

Open `notebooks/srq_generalization_m11_adaptive_precision_colab.ipynb` in a
fresh T4 runtime, execute cells in order, and upload the exact M6 ZIP when
requested. The notebook creates only `train.pt`, refuses a visible `test.pt`,
and downloads:

`srq_generalization_m11_adaptive_precision_train_only.zip`

Return the ZIP whether the formal status is PASS or FAIL.

## Permitted conclusion

A PASS supports a narrow claim that one preregistered, factor-only adaptive
precision rule repairs the observed M6 10k/20k retention gate while retaining
at least 75% total-state reduction. It does not prove optimal block selection,
generalization to GACL's mini-batch-frequency compression, or universal
superiority over fixed INT8 or FP16.
