# M24 protocol: equal-memory packed and low-rank controls

## Question

M23 compares SRQ-INT8 with a dense FP32 Exact learner, a narrower dense
Exact learner, CountSketch and raw-feature Ridge.  A reviewer can still ask
two narrower questions:

1. Does the same-memory comparison change when an Exact symmetric matrix is
   physically stored without duplicating its lower triangle?
2. Does a full-width streaming low-rank summary outperform SRQ at the same
   persistent-state budget?

M24 answers only these two questions.  It is a train-only validation study
and does not authorize reading, creating or selecting from a test cache.

## Frozen source experiment

The source artifact is
`srq_generalization_m23_equal_budget_multistream_train_only.zip`, SHA-256
`80c8940a1d7ff11fd71b14c127802ca1a80bc76aec608cbd93d42fc8a8595b2a`.
M24 requires its PASS status, its three streams `s2025`, `s2026`, `s2027`,
and its train/projection/split identities.  The byte ceiling is the M23
full-width SRQ-INT8 final persistent state: 91,880,088 bytes.  The ceiling is
read and locked before any representation is encoded or any accuracy is
computed.

The cached ViT-B/16 features, class orders, outer train/validation splits,
calibration indices and 10,000-dimensional RanPAC projections must match the
corresponding M23 stream.  Each stream seed controls all of those stochastic
objects exactly as in M23.

## E1a: packed Exact

Packed Exact retains every unique entry of the symmetric FP32 Gram matrix,
plus the cross statistic, class counts, classifier and random projection.  It
stores diagonal blocks as upper-triangular vectors and off-diagonal blocks
once.  It reconstructs a dense symmetric matrix only as an update workspace;
that workspace is discarded before `update` returns and is not counted as
persistent state.

The expanded dimension is the largest integer not exceeding 10,000 for which

```
projection + packed Gram + cross + counts + classifier <= 91,880,088 bytes.
```

This rule gives width 5,878 and 91,873,540 bytes.  Width 5,879 must exceed the
ceiling.  Its projection is the first 5,878 columns of the verified M23
10,000-dimensional projection, so it is a nested-width control rather than a
new random draw.  Ridge lambda is selected independently for this representation by
the same train-only MSE calibration protocol and candidate grid as M23.  No
accuracy enters the width calculation.

## E1b: Frequent-Directions Ridge

FD-Ridge keeps the original 10,000-dimensional random-ReLU representation.
It replaces the dense Gram by a deterministic FP32 Frequent Directions
summary `B` and solves

```
(B^T B + lambda I) W = Q
```

with the Woodbury identity.  The sketch is updated with the standard
`sigma_ell^2` shrinkage and never retains an input row after an update.  The
rank is the largest integer for which

```
full projection + B + cross + counts + classifier <= 91,880,088 bytes.
```

The classifier is stored as a compact FP64 Woodbury correction of shape
`rank x classes`, not as a dense FP32 10,000-by-class weight matrix; these
FP64 bytes are included in the accounting. This avoids cancellation in the
wide Woodbury subtraction. The rule gives rank 1,400 and 91,840,400 bytes.
Rank 1,401 must exceed the ceiling. Because FD uses exactly the M23 full random-ReLU representation, it
uses M23's preregistered full-representation Ridge lambda for the same stream;
accuracy is not used to retune the low-rank method.

## Measurements

For both controls and every task, record validation accuracy, persistent
tensor bytes, solver relative residual and cumulative analytic update time.
Report AIA and final validation accuracy per stream, then mean and sample
standard deviation over the three streams.  Also report paired AIA
differences against M23 SRQ-INT8.  These differences are measurements, not
completion gates.

## Validity gates

M24 passes only if all of the following structural gates pass:

- the exact M23 archive and all three source streams are verified;
- no `test.pt` is visible and `uses_test_set` is false;
- train cache, class order, split, calibration and projection identities
  match M23 for every stream;
- the byte locks are identical across streams and are computed before any
  accuracy;
- final physical tensor bytes equal symbolic accounting and do not exceed the
  ceiling;
- adding one width or one sketch rank would exceed the ceiling;
- all updates are finite, exemplar-free and have solver relative residual at
  most `1e-4`;
- all three stream outputs are complete.

There is deliberately no gate on AIA, final accuracy, or whether SRQ wins.
Negative results remain valid experimental results.

## Output and interpretation boundary

The runner writes an atomic JSON per stream and a combined
`m24_results.json`.  Existing complete stream files may be reused only when
their configuration, runner, backend and source-artifact identities match.
Feature tensors are excluded from the exported archive.

M24 supports a claim about persistent learner state, not peak workspace or
wall-clock superiority.  Packed Exact can require a dense temporary solve
workspace, and Frequent Directions can be computationally expensive because
of repeated SVDs.  Both costs must be reported rather than hidden.
