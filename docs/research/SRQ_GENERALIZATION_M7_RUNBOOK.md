# SRQ generalization M7 task-wise error runbook

Status: `PASS_M7_ERROR_TRAJECTORY_TRAIN_ONLY`; the CIFAR-100 train-only
diagnostic is complete. M7 does not authorize test evaluation, precision
tuning, or post-hoc width selection.

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

## Recorded result

Artifact SHA-256:
`df92adadce046c53efa5b9fcf01435d1fab2a4c71a690b10c3d7c221205acf36`.
Result JSON SHA-256:
`1799ad863ab389f67b13ffdb59df55b7dc436d217e5e54c45ad28e6d4eaa0bf8`.
The ZIP passes CRC, contains exactly the four expected files, embeds the
byte-identical config, reports clean source commit `dd92292`, and has
`uses_test_set=false`. Its 60 CSV rows match all JSON scalar values, the SVG
parses successfully, and all task accuracies are exactly identical to the M6
records at widths 10k and 20k.

| Final task diagnostic | Width 10k | Width 20k | 20k / 10k |
|---|---:|---:|---:|
| Local factor error | 0.006043 | 0.006140 | 1.016x |
| System-action error | 0.008589 | 0.009021 | 1.050x |
| Weight error | 0.027216 | 0.048829 | 1.794x |
| Logit error | 0.317845 | 0.635150 | 1.998x |
| Prediction-change fraction | 1.54% | 2.76% | 1.792x |
| Margin-certified fraction | 84.19% | 75.43% | -- |
| Exact--P2B final accuracy | 0.22 pp | 0.40 pp | -- |
| Exact--P2B AIA | 0.0877 pp | 0.2558 pp | -- |

P2B weight and logit errors increase on all nine transitions at both widths.
The system-action error increases on eight of nine transitions. In contrast,
FP16 final system-action error stays near `2.7e-4`, final prediction agreement
is at least `99.93%`, and its final accuracy gap is at most `0.02` pp. The
Pearson correlations between P2B system-action error and per-task accuracy gap
are 0.686 at 10k and 0.865 at 20k.

This supports the interpretation that repeated INT8 state approximation is
associated with cumulative effective-system drift and stronger downstream
amplification at width 20k. It does not prove causality: task index is a common
driver, the system error is a 16-vector randomized action estimate rather than
the full matrix norm, and the experiment does not estimate eigenvalues or a
condition number.
