# SRQ-FLY Priority 2C: implicit Ridge initialization

## Decision being tested

Priority 2B selected streaming factor quantization with 64 blocks per batch.
Its whole-update CUDA peak is no longer caused by quantization; the remaining
maximum occurs on task one while forming `Z.T @ Z + lambda*I` and running dense
Cholesky.

Priority 2C tests one final implementation-only change.  The Ridge system before
any observation is `lambda*I`, so its upper square root is

\[
R_0=\sqrt{\lambda}I.
\]

Instead of materializing the first Gram and invoking Cholesky, the candidate
applies the existing blocked QR update to `[R_0; Z_1]`.  In exact arithmetic,

\[
R_1^T R_1=R_0^T R_0+Z_1^T Z_1
=\lambda I+Z_1^T Z_1.
\]

All later updates, batch-64 streaming quantization, checkpoint tensor format,
classifier solve and inference are unchanged.  The first-update backend is
recorded in the checkpoint and cannot silently change on resume.

## Locked isolated benchmark

The benchmark is synthetic-only on a Tesla T4: seed 2025, width 10,000, two
updates, one warm-up and seven measured repetitions.  Each method/repetition
runs in a fresh process, and method order rotates by repetition:

1. dense Exact FLY;
2. locked Priority-2B streaming batch-64 with first-task Gram/Cholesky;
3. streaming batch-64 with implicit-Ridge first-task blocked QR.

The candidate passes only when all of the following hold:

- relative logit drift from Priority 2B is at most `1e-5`;
- persistent tensor bytes are exactly unchanged;
- solver relative residual is at most `1e-5`;
- median paired update-time ratio to Priority 2B is at most `1.10`;
- median paired peak-allocated ratio to Priority 2B is at most `0.90`.

The last threshold requires a material peak-memory reduction; merely changing
the numerical path is not sufficient.

## Stop boundary

`PASS_REVIEW_PRIORITY2C` establishes only synthetic CUDA allocation and
predictor feasibility.  It does not use a dataset and does not authorize a
held-out test.  A pass locks the final optimized backend for a real CIFAR-100
train-only equivalence run.  A failure retains Priority 2B batch-64 as the final
backend; gates must not be relaxed after observing output.

Packed int4 and error feedback are explicitly outside this optimization path.
They define new accuracy/state trade-offs and require separate ablations, so
they do not block the final SRQ-FLY evaluation.
