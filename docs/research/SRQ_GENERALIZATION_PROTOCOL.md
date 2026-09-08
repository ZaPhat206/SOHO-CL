# SRQ analytic-backend generalization protocol

Status: active staged protocol. This document does not authorize a held-out
test run. A later milestone may begin only after the preceding gate has a
recorded PASS result.

## Research question

The study asks whether structure-preserving square-root compression is a
reusable backend for high-dimensional analytic continual learners, rather
than an optimization specific to FLY. The admissible learner family has a
fixed explicit feature map for historical samples and additive Ridge
sufficient statistics

\[
A_t=\Lambda+\sum_{k\le t}\Phi_k^\top\Phi_k,\qquad
B_t=\sum_{k\le t}\Phi_k^\top Y_k,\qquad
W_t=A_t^{-1}B_t.
\]

The protocol does not claim compatibility with a changing historical feature
map, prototype-only classifiers, iterative softmax training, covariance
downdates, negative weights, or inverse-RLS implementations without a
separate mathematically equivalent adapter.

## Frozen current method

Until the FLY regression gate passes, the reference compressed method is P2B:

- mixed INT8/FP32 upper square-root storage;
- FP32 diagonal and groupwise-INT8 strict upper triangle;
- block size 256 and group size 64;
- blocked QR with panel size 128;
- first update by Gram--Cholesky;
- streaming quantization in batches of 64 blocks.

No precision allocation, error feedback, packed INT4, Ridge retuning, or
checkpoint-format change is allowed during backend extraction.

## Comparison contract

Within one comparison, every backend receives byte-identical `Phi` and labels
and uses the same class order, task grouping, Ridge coefficient, dtype, seed,
and evaluation schedule. Method-specific hyperparameters are selected using
train-only validation. Test accuracy cannot choose a backend, threshold,
width, rank, precision policy, or stopping point.

Persistent bytes include the projection when it is deployed with the
frontend, the quadratic state, quantization scales and metadata tensors,
cross statistic, class counts, and classifier. Feature/code caches are
reported separately and are never treated as learner state.

## Milestones and gates

### M0 -- evidence and source lock

Work:

1. record the current source, config, artifact, test-use, and caveat identities;
2. exclude logs and local experiment caches from source control;
3. synchronize the manuscript results ledger with the immutable evidence;
4. run the focused SRQ regression suite and the full repository suite.

Gate: the machine-readable manifest parses, every referenced repository file
exists, document consistency checks pass, and the exact test results are
recorded. A dirty unrelated worktree is allowed for local documentation only;
all experiment workers must later run from a clean committed clone.

### M1 -- generic backend extraction

Work: extract class expansion, cross-statistic update, Exact Gram, square-root
update, compressed storage, state accounting, checkpointing, and diagnostics
under `methods/analytic_ridge/`. FLY projection and WTA remain in a frontend
wrapper.

Gate: legacy and extracted P2B produce byte-identical compressed blocks,
classifier tensors within the locked numerical tolerance, identical
persistent bytes, and compatible checkpoint round trips on synthetic streams.

Required tests (once created):

```text
python -m pytest -q tests/test_analytic_ridge_backend.py
```

### M2 -- unquantized equivalence

Work: compare Exact Gram, dense QR, and blocked QR in FP64 and FP32. Record
factor/system error, weight error, solver residual, logit drift, and prediction
agreement after every task.

Gate: FP64 blocked QR matches dense QR and Exact Gram at numerical tolerance;
FP32 square-root introduces no material prediction or accuracy change on the
declared train-only development stream.

### M3 -- FLY train-only regression (PASS)

Work: compare legacy and generic Exact/P2B learners on the same cached CIFAR
training stream.

Gate: representation, target statistics, selected hyperparameters, persistent
bytes, and predictions satisfy the preregistered regression tolerances. This
is the earliest milestone that may require a Colab GPU run.

Recorded artifact: `srq_generalization_m3_fly_regression.zip`, SHA-256
`ec376875ff0e29e2d7a46d1fcb9fcccc523ea47fbe6a8f2dd83ec3a98930a595`.
All ten task records pass exact legacy/generic tensor and byte identity.

### M4 -- RanPAC adapter and train-only gate (PASS)

Work: reproduce the original analytic RanPAC head, then compare original,
generic Exact Gram, FP32 square-root, FP16 square-root, and fixed P2B SRQ on
identical expanded features.

Gate: original and generic Exact semantics match; FP32 square-root matches
Exact; fixed SRQ reduces the quadratic persistent state by at least 70%, has
at most 0.20 percentage-point validation-AIA loss, and does not show a
numerical failure. Failure keeps the paper scoped to SRQ-FLY and cannot be
overridden using test accuracy.

The source-locked run instructions and the distinction between the official
per-task Ridge schedule and the controlled fixed-Ridge comparison are in
`docs/research/SRQ_GENERALIZATION_M4_RUNBOOK.md`.

Recorded artifact: `srq_generalization_m4_ranpac_train_only.zip`, SHA-256
`228da0828c7f6964bcc8f7da92258efa20b00eef0fc83679246ce0ae2a0c8f52`.
All eight gates passed on the locked CIFAR train-only stream. This closes the
initial non-FLY frontend gate but remains one controlled development seed.

### M5 -- equal-budget alternatives (PASS)

Work: compare full-width Exact, byte-matched reduced width, FP16 square-root,
fixed SRQ, raw-feature Ridge, and a fixed signed-hash feature-sketch Ridge.
Sketch dimensions and frontend widths are derived from total persistent byte
budgets before accuracy is read. The source-locked procedure is specified in
`docs/research/SRQ_GENERALIZATION_M5_RUNBOOK.md`.
The previously preregistered low-rank/streaming Ridge sketch slot is therefore
instantiated by an explicit CountSketch feature map followed by Exact Ridge;
it is not mislabeled as a factor-space approximation of the full Gram.

Gate: SRQ lies on or close to the observed accuracy--persistent-state Pareto
frontier. If a sketch dominates SRQ in accuracy, state, and update cost, the
claim must be narrowed before proceeding.

Recorded artifact: `srq_generalization_m5_equal_budget_train_only.zip`,
SHA-256
`0f5e23fa4a7d83926638641025fb103895561073cd1f0fa4e1e0d552fd2fa931`.
All seven gates passed. At approximately 91.9 MB total persistent state, P2B
exceeded reduced-width Exact and CountSketch Exact by 0.4838 and 0.5720
validation-AIA points, respectively. P2B was not Pareto dominated by the five
tested alternatives, but its analytic update was 15.25 and 19.22 times slower
than those two equal-budget controls. This remains one train-only development
seed and does not establish a global Pareto frontier.

### M6 -- width scaling (formal FAIL; diagnostic complete)

Work: sweep fixed widths 2k, 4k, 6k, 8k, 10k, 15k, and 20k for Exact Gram, FP16
square-root, and fixed P2B on the controlled RanPAC train-only stream. Widths
are reporting points, not accuracy-selected candidates. Projection prefixes,
splits, and per-width Ridge calibration are shared across the three backends.
The source-locked procedure and plot contract are in
`docs/research/SRQ_GENERALIZATION_M6_RUNBOOK.md`.

Gate: expose the empirical accuracy--state--update curves, preserve numerical
and accuracy-retention bounds at every width, and retain the quadratic state
terms in the report.

Recorded artifact: `srq_generalization_m6_width_sweep_train_only.zip`,
SHA-256
`b2739b9da023ebd2eedb6fdfe01c394e94f252773e847533b35350021c3d239e`.
The artifact is complete, train-only, source-clean, and numerically stable,
but its formal status is `FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY`. At width 20,000,
P2B loses `0.255845` validation-AIA points to Exact, exceeding the locked
`0.25`-point bound by `0.005845` points. The threshold is not changed after
observing the result. FP16 loses at most `0.0100` points and the maximum solver
residual is `5.83e-6`, which localizes the observed degradation to INT8 state
approximation rather than failure of the square-root update or linear solve.
M7 is therefore a diagnostic follow-up, not a retroactive M6 rescue.

### M7--M9 -- error, theory, and systems

- task-wise factor/system/solution/logit error and prediction agreement;
- exact square-root identity, structural-SPD statement, factor-to-system and
  Ridge-solution perturbation bounds, and a margin-preservation corollary;
- at least three isolated systems repetitions separating persistent bytes,
  checkpoint bytes, PyTorch allocated/reserved peaks, process NVML peak, and
  per-stage time.

Gate: claims and plots expose the remaining quadratic scaling and slower
update rather than hiding them.

M7 must retain widths 10,000 and 20,000 as predeclared diagnostic points and
must not select a replacement width or precision from validation accuracy. It
measures task-wise quantization error, randomized reconstructed-system action
error, solution and logit drift, prediction agreement, and classification
margins against Exact and FP16 references. Its purpose is to test whether the
widening M6 gap is consistent with accumulated INT8 perturbation while the
solver remains stable.

### M10 -- optional GACL generalized-stream adapter

GACL is attempted only after the equal-budget gate. Its inverse-RLS recurrence
must first be shown equivalent to the additive primal Ridge system on streams
containing both exposed and unexposed classes. Its original exact/joint weight
invariance is distinguished from the controlled approximation introduced by
quantization. Because the official setting updates by mini-batch, compression
frequency and anytime-evaluation semantics require a separate locked adapter;
they cannot silently reuse the task-level FLY schedule.

### M11 -- optional adaptive precision

Budget-aware FP16/INT8 block allocation is attempted only if fixed INT8 shows
frontend-dependent error and FP16 recovers it. Precision masks and all scale
metadata count toward persistent state. Packed INT4 and error feedback remain
later alternatives, not prerequisites.

### M12 -- final locked evaluation

Final evaluation is authorized only after configurations, budgets, seeds,
statistics, and source hashes are committed. It has no accuracy gate, no
test-time retry, and no post-test method selection.

## Naming gate

- FLY evidence only: `SRQ-FLY`.
- FLY plus a passing RanPAC integration: a reusable backend demonstrated on
  two analytic learners.
- FLY, RanPAC, and a passing independent GACL adapter: a generic backend for
  the explicitly defined fixed-feature additive-Ridge family, demonstrated in
  both class-disjoint and generalized streams.

The term `universal plug-in` is prohibited.
