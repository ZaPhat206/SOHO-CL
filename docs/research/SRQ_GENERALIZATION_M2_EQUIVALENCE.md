# SRQ generalization M2 unquantized equivalence

Date: 2026-09-07

## Outcome

M2 separates the numerical effect of the square-root update from the effect of
quantization. Three unquantized backends received identical feature rows and
labels after every task:

- Exact Gram accumulation followed by a dense Ridge solve;
- a dense QR update of the upper square-root factor;
- the production blocked-QR update, without factor quantization.

The audit used deterministic Gaussian, column-scaled, and sparse streams in
both FP64 and FP32. It recorded reconstructed-system error, classifier-weight
error, solver residual, logit drift, prediction agreement, and the direct
blocked-versus-dense factor difference after every task. It used no dataset,
feature cache, validation accuracy, or test split.

## Locked protocol

Source commit: `1f7354c350cff5c7a45288310472e07dc67d32b2`.

Configuration:
`configs/srq_generalization_m2_equivalence.json`, SHA-256
`d3ebbbccde667c5140aed53b317eb3b132ef672f6879c40c2a0bc433fd661e8b`.

The stream has dimension 64, five tasks, 41 rows and three new classes per
task, Ridge coefficient 100, and a fixed 257-row prediction probe. Blocked QR
uses panel size 13 and trailing chunk size 17 so the test exercises partial
panels and chunks rather than only evenly divisible dimensions.

## Numerical evidence

The table reports the maximum over all five tasks and over dense and blocked
QR where applicable.

| Stream | Dtype | Relative system error | Relative weight error | Solver residual | Relative logit error | Minimum prediction agreement |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian | FP64 | 3.43e-16 | 2.11e-16 | 3.28e-16 | 6.24e-16 | 100% |
| Column-scaled | FP64 | 4.51e-16 | 3.26e-16 | 4.59e-16 | 1.97e-15 | 100% |
| Sparse | FP64 | 3.67e-16 | 1.33e-16 | 1.73e-16 | 4.69e-16 | 100% |
| Gaussian | FP32 | 1.80e-7 | 1.13e-7 | 1.83e-7 | 3.48e-7 | 100% |
| Column-scaled | FP32 | 1.32e-7 | 1.24e-7 | 1.89e-7 | 9.62e-7 | 100% |
| Sparse | FP32 | 1.95e-7 | 7.32e-8 | 9.64e-8 | 2.40e-7 | 100% |

The largest blocked-versus-dense factor difference was `5.04e-16` in FP64
and `2.22e-7` in FP32. All preregistered gates passed without changing their
thresholds.

## Interpretation and effect

In exact arithmetic, QR of the stacked matrix
`[R_(t-1); Phi_t]` and direct Gram accumulation represent the same updated
system because

\[
R_t^\top R_t=R_{t-1}^\top R_{t-1}+\Phi_t^\top\Phi_t.
\]

M2 shows that the implementation retains this identity to ordinary floating
point rounding error. Block partitioning changes the order of operations but
does not introduce a material classifier or prediction difference in the
declared stress cases. Consequently, later Exact-versus-SRQ drift must be
measured as a quantization/mixed-precision effect on top of this verified QR
baseline. M2 does not establish behavior for arbitrary condition numbers,
larger dimensions, or quantized factors; those remain later empirical and
theoretical milestones.

## Validation

Focused command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_analytic_ridge_equivalence.py tests/test_analytic_ridge_backend.py
```

Result: `12 passed, 1 warning in 6.45s`.

Full repository command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `490 passed, 20 warnings in 105.22s`. The warnings were the existing
PyTorch JIT deprecations and sparse tensor notices; no test failed.

Source-locked audit command:

```powershell
python -u tools/srq_generalization_m2.py run `
  --config configs/srq_generalization_m2_equivalence.json `
  --output tmp/srq_generalization_m2_results.json
```

Result: `PASS_M2_UNQUANTIZED_EQUIVALENCE`. The warning in the focused test is
the existing PyTorch sparse-CSC beta notice from the legacy FLY compatibility
test; M2 itself does not use a sparse PyTorch matrix.

## Gate decision

`PASS_M2_UNQUANTIZED_EQUIVALENCE`.

M3 FLY train-only regression is authorized. Held-out evaluation remains
unauthorized.
