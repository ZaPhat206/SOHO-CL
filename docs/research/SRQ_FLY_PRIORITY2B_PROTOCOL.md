# SRQ-FLY Priority 2B: streaming factor quantization

## Motivation

The immutable Priority-2A artifact
`srq_fly_priority2a_memory.zip` (SHA-256
`8ec01f49ff5a8a5767f54e29a9c5a3197461de6d67283c4db3d30477019c4aaf`)
returned `STOP_MEMORY_GATE`.  Chunking reduced the task-2 blocked-QR temporary
from 84.4 MiB to as little as 5.0 MiB, but every candidate retained a 1.850 GiB
whole-update peak.  The profiled factor-quantization stage added approximately
670 MiB because the eager encoder retained extracted floating-point values for
all strict-upper blocks.

Priority 2B releases dead Gram/Cholesky-input buffers before factor encoding
and changes allocation scheduling inside the version-1 encoder.  It does not
change projection, WTA, Ridge, QR, quantization formula, scales, factor bytes,
cross statistic, classifier, or inference.

## Candidate encoder

The eager encoder stores descriptors of the form `(row, column, values)`, where
`values` may own a contiguous copy of a non-contiguous matrix block.  The
streaming encoder stores only `(row, column, length)`.  For one bounded batch it:

1. extracts at most `b` blocks of equal length;
2. applies the unchanged per-block groupwise symmetric int8 quantizer;
3. writes the decoded values back into the disposable factor;
4. accumulates the diagnostic error; and
5. releases floating-point temporaries before reading the next batch.

Whole-matrix finite validation is likewise performed in bounded row chunks.
The locked batch grid is `{1,4,16,64}`.  Unit tests require stored int8 values,
float32 scales, reconstructed factors, weights, state bytes, and logits to
match the eager encoder.

## Repeated isolated benchmark

The study is synthetic-only and uses seed `2025`, width 10,000, two updates,
one warm-up round, and seven measured rounds on a Tesla T4.  Every
method/repetition runs in a fresh process; method order rotates by round.  The
controls are:

1. dense Exact FLY;
2. eager in-place SRQ quantization with consuming blocked QR and dead-buffer
   cleanup;
3. four streaming quantization batch sizes.

The first measured eager/streaming run performs a second, untimed stage
profile.  Before this profile, the timed learner is deleted and the CUDA cache
is cleared so absolute stage peaks do not include unrelated learner state.

## Locked gates and selection

The eager-cleanup control and every streaming candidate are eligible only when:

- maximum relative logit drift from eager `<=1e-5`;
- persistent tensor bytes equal eager exactly;
- maximum solver relative residual `<=1e-5`;
- median paired update-time ratio to eager `<=1.10`;
- median paired peak-allocated ratio to Exact FLY `<=1.05`.

Among eligible candidates, select minimum median peak ratio, then minimum
median time ratio, prefer the simpler eager encoder on an exact tie, then use
smaller batch size.  If no candidate passes, return
`STOP_QUANTIZATION_GATE`; do not relax thresholds after observing results.

## Claims boundary

A pass establishes allocation feasibility only on the recorded CUDA/software
stack.  It neither uses nor authorizes train/test evaluation.  The selected
backend must next pass a train-only predictor-equivalence study before it may
replace the Priority-1 implementation.  If Priority 2B fails, the preregistered
next target is the first-task Gram/Cholesky path via an implicit ridge factor.
