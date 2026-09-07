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

### M4 -- RanPAC adapter and train-only gate (implementation ready)

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

### M5 -- equal-budget alternatives

Work: compare full-width Exact, byte-matched reduced width, FP16 square-root,
fixed SRQ, raw-feature Ridge, and a low-rank/streaming Ridge sketch. Ranks and
widths are derived from byte budgets before accuracy is read.

Gate: SRQ lies on or close to the observed accuracy--persistent-state Pareto
frontier. If a sketch dominates SRQ in accuracy, state, and update cost, the
claim must be narrowed before proceeding.

### M6--M9 -- scaling, error, theory, and systems

- width sweep: 2k, 4k, 6k, 8k, and 10k dimensions;
- task-wise factor/system/solution/logit error and prediction agreement;
- exact square-root identity, structural-SPD statement, factor-to-system and
  Ridge-solution perturbation bounds, and a margin-preservation corollary;
- at least three isolated systems repetitions separating persistent bytes,
  checkpoint bytes, PyTorch allocated/reserved peaks, process NVML peak, and
  per-stage time.

Gate: claims and plots expose the remaining quadratic scaling and slower
update rather than hiding them.

### M10 -- optional second external analytic learner

ACIL is attempted only after RanPAC passes. Its original exact/joint
equivalence is distinguished from the controlled approximation introduced by
quantization. F-OAL requires a separate square-root RLS derivation and is not
part of the initial plug-in claim.

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
- FLY, RanPAC, and a passing independent ACIL adapter: a generic backend for
  the explicitly defined additive-Ridge family.

The term `universal plug-in` is prohibited.
