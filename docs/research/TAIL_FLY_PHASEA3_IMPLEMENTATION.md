# TAIL-FLY Phase A3 numerical audit

Status: implementation complete; the locked ImageNet-R train-only Colab run is
pending.

## Why A3 exists

Phase A was a valid negative result: the selected TAIL model was 6.5213
percentage points below matched exact FLY and 4.6127 points below raw Ridge.
It also reported a maximum analytic-solver relative residual of
`3.1599e-3`, above the locked `1e-5` threshold. A3 separates numerical error
from approximation error without changing the seed, representation, ranks,
Ridge grid, split, backbone, or cached WTA codes.

## Locked changes

- streamed SVD, diagonal, and cross-statistics remain `float32`;
- only the Woodbury/TSVD/diagonal classifier solves and their residual checks
  run in `float64`;
- residuals are recorded for every method, rank, Ridge value, and task;
- resumable unit identity includes the Git commit and SHA-256 of the runner,
  solver, learner, and streaming-SVD sources;
- a new gate compares selected TAIL against the independently best selected
  plain TSVD, in addition to the matched rank/Ridge control;
- the held-out ImageNet-R test feature file must remain absent.

The locked configuration is
`configs/tail_fly_imagenetr_train_only_a3.json`. Its SHA-256 is
`24f2d82b5e5662bbc315bbff46cbb962f92ea7bdc41ad9cb7e2ff6d77c129b96`.

## Interpretation boundary

If numerical stability passes but accuracy remains below the independently
selected plain TSVD, raw Ridge, or exact FLY gates, the deficit is evidence
against the approximation rather than evidence of a solver bug. A3 does not
authorize a new search grid, seed change, dataset change, or held-out test run.

## Verification

Targeted command:

```text
python -m pytest -q tests/test_tail_fly_math.py tests/test_tail_fly_learner.py tests/test_tail_fly_phasea.py
```

Result: `24 passed`.

Full repository command:

```text
python -m pytest -q
```

Result: `193 passed`. Warnings were limited to the existing PyTorch sparse-CSC
beta/invariant notices and torch JIT deprecation notices.
