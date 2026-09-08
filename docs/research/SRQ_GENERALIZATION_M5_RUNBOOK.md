# SRQ generalization M5 equal-budget runbook

Status: implementation ready; the real CIFAR-100 train-only GPU gate has not
yet been run. M5 does not authorize a held-out test evaluation.

## Question

M5 tests whether fixed P2B remains on, or sufficiently close to, the observed
accuracy--persistent-state Pareto frontier once simpler equal/lower-state
alternatives are included. It uses the same controlled RanPAC Phase-2
random-ReLU path and the same outer train/validation split as M4.

The six methods are:

1. Exact Gram at the full expansion width 10,000;
2. FP16 square-root at width 10,000;
3. frozen P2B mixed INT8/FP32 at width 10,000;
4. Exact Gram with its width reduced to the largest integer fitting the P2B
   total persistent-state budget;
5. full-width random-ReLU followed by a fixed one-nonzero signed CountSketch
   and Exact Ridge at the largest sketch dimension fitting the same budget;
6. Exact Ridge directly on the frozen 768-dimensional ViT features.

CountSketch is an explicit lower-dimensional feature-sketch control. It is not
described as square-root compression or as a low-rank approximation of the
same full-rank Gram.

## Pre-accuracy byte lock

Dimensions are derived from tensor shapes and dtypes before any representation
is encoded or validation accuracy is available. Persistent bytes include the
deployed random projection, Gram/factor values, quantization scales, class
counts, cross statistic, classifier, and CountSketch index/sign tensors where
applicable.

For the locked 768-to-10,000 frontend and 100 final classes:

- P2B budget: `91,880,088` bytes;
- byte-matched Exact width: `4,333`, using `91,877,332` bytes;
- CountSketch dimension: `3,809`, using `91,851,524` bytes;
- raw-feature Exact Ridge: `2,974,096` bytes.

The reduced-width projection is the first 4,333 columns of the locked
full-width projection. CountSketch assigns every one of the 10,000 expanded
coordinates to one bucket with a deterministic Rademacher sign using seed
2025. Neither alternative uses labels or accuracy to choose its dimension.

## Ridge selection

Full-width Exact, FP16 and P2B share one Ridge coefficient because they receive
the same expanded representation. Reduced-width Exact, CountSketch and raw
Ridge each select their own coefficient from the same locked grid using the
same disjoint calibration subset of the outer training partition. Selection
uses one-hot mean-squared error and the smallest-lambda tie break. The outer
validation partition never selects a dimension, rank, precision mode or Ridge
coefficient.

## Gates

- both equal-budget dimensions are locked before accuracy and underfill the
  P2B byte budget by no more than 0.1%;
- P2B loses at most 0.20 percentage points validation AIA to full-width Exact;
- every solver residual is at most `1e-5`;
- no equal/lower-state alternative exceeds P2B validation AIA by more than
  0.20 percentage points;
- no alternative simultaneously has no worse validation AIA, final validation
  accuracy, persistent bytes, and analytic-update time, with at least one
  strict improvement.

If the last two gates fail, the artifact is still scientifically useful, but
the manuscript must narrow the Pareto claim. Test accuracy cannot override a
failure.

## Run on Colab

After committing and pushing the source-locked implementation, open and run
every cell in:

`notebooks/srq_generalization_m5_equal_budget_colab.ipynb`

The notebook creates only a CIFAR training-feature cache, verifies that no
`test.pt` exists, runs the focused synthetic suite, executes the M5 gate and
exports:

`srq_generalization_m5_equal_budget_train_only.zip`

The archive must contain exactly `m5_results.json` and the byte-identical
locked config. Do not include the feature cache, projected features, model
checkpoint or any test tensor.

## Interpretation

A PASS supports the narrow claim that P2B is not displaced by the tested
reduced-width, signed-hash sketch or raw-feature alternatives on this locked
train-only stream. It does not prove global Pareto optimality over every Ridge
sketch, quantizer, width or dataset. A later systems milestone must repeat
runtime measurements before timing becomes a broad claim.
