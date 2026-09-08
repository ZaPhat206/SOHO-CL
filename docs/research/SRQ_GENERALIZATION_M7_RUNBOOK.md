# SRQ generalization M7 task-wise error runbook

Status: implementation ready; the real CIFAR-100 train-only diagnostic has
not yet been run. M7 does not authorize test evaluation, precision tuning, or
post-hoc width selection.

## Why M7 exists

M6 formally failed because the Exact--P2B validation-AIA gap at width 20,000
was 0.255845 percentage points, 0.005845 points beyond the locked 0.25-point
limit. At the same time, FP16 remained within 0.0100 points of Exact and every
solver residual remained below `5.83e-6`. M7 therefore tests the narrower
mechanistic hypothesis that fixed INT8 state approximation, rather than
blocked QR or the triangular solver, explains the widening gap.

The M6 threshold is not changed. M7 is a diagnostic PASS/FAIL based on
provenance, completeness, finite measurements, and numerical integrity; it
has no accuracy-retention threshold and cannot rescue M6.

## Locked comparison

- frontend: controlled RanPAC Phase-2 standard-normal random projection plus
  ReLU, without PETL;
- dataset view: CIFAR-100 train partition only;
- diagnostic widths: 10,000 and 20,000;
- Ridge coefficient: `1e6` at both widths, verified against the immutable M6
  artifact;
- methods: Exact FP32 Gram, square-root FP16, and P2B INT8/FP32;
- ten class-incremental updates with the same seed, class order, partitions,
  projection prefixes, and evaluation schedule as M6;
- sixteen fixed Rademacher probe vectors per width, generated independently
  of labels and accuracy; the 10k signs are the row prefix of the 20k probe
  matrix and each width is normalized by its square root.

The notebook requires the exact M6 ZIP with SHA-256
`b2739b9da023ebd2eedb6fdfe01c394e94f252773e847533b35350021c3d239e`.
The runner verifies both the archive and embedded result identity before
loading the training cache.

## Recorded diagnostics

Let `A_t` be the Exact Ridge system, `Rhat_t` the decoded compressed factor,
`V` the fixed probe matrix, `W_t` the Exact weights, and `What_t` the
compressed-backend weights. M7 records after every task:

- local factor quantization error reported while encoding the current factor;
- randomized relative system-action error
  `||Rhat_t^T Rhat_t V - A_t V||_F / max(||A_t V||_F, 1)`;
- relative classifier-weight error `||What_t-W_t||_F/max(||W_t||_F,1)`;
- relative validation-logit error;
- prediction agreement and Exact-minus-method accuracy gap;
- Exact top-1 margin and the fraction satisfying
  `2 ||logit_error_i||_infinity < exact_margin_i`.

The system-action metric is a fixed randomized diagnostic, not the full
Frobenius norm of the 20k-by-20k system. It avoids materializing an additional
dense comparison matrix while retaining sensitivity to cumulative system
drift. The margin inequality is a sufficient certificate for an unchanged
top-1 prediction; samples outside it may still agree.

## Gates

- the source M6 archive and embedded result match their locked SHA-256 values;
- both widths and all ten tasks complete for all three methods;
- the independently accumulated Exact system action matches direct Exact Gram
  action to relative error at most `1e-5`;
- every solver relative residual is at most `1e-5`;
- every reported diagnostic is finite.

No gate requires monotonic error growth or a particular correlation with
accuracy. Those are empirical outcomes, not mathematical necessities.

## Expected output

The Colab notebook exports:

- `config.json`;
- `m7_results.json`;
- `error_trajectory.csv`;
- `m7_error_trajectory.svg`.

Feature caches, the backbone checkpoint, the M6 input artifact, and all
sample-level tensors are excluded from the evidence ZIP.

## Allowed conclusion

If system, weight, and logit errors grow together while FP16 remains near
Exact and solvers stay stable, the result is consistent with cumulative INT8
perturbation. It is not by itself a causal proof. If the diagnostics do not
track the accuracy gap, the paper must say that M6 reveals a width-dependent
accuracy limit whose mechanism remains unresolved.
