# TAIL-FLY Phase A implementation record

Status: implementation/correctness phase complete locally. The ImageNet-R
train-only Colab development run has not started.

## Implemented scope

- isolated `methods/tail_fly` package; no edits to SOHO, FlyCL, CARS-FLY, or
  other method implementations;
- continual QR/core-SVD recurrence over full post-Top-K FLY codes;
- exact coordinate second moments, exact label cross-statistic, counts, and
  dynamic global class mapping;
- low-rank-plus-diagonal Woodbury Ridge solve with Cholesky and no explicit
  inverse;
- plain TSVD-FLY and diagonal-only matched controls;
- exact matched FLY and raw-Ridge train-only controls in a strict runner;
- verified, resumable WTA experiment cache that is explicitly excluded from
  learner state;
- checkpoint round-trip that serializes only the fixed projection and
  aggregate statistics, then rebuilds the classifier;
- a locked ImageNet-R development config and Colab notebook.

The production ImageNet-R configuration uses float32 statistics so the
10,000-dimensional QR/SVD path is practical on a T4. Mathematical equivalence
tests use float64. The same `1e-5` residual gate remains locked for production.

At final `C=200`, `m=10000`, rank 256, and float32, the analytic resident-state
estimate is about 62.3 MB including the fixed sparse projection and derived
classifier, versus about 452 MB for matched exact FLY. This estimate is not a
measured peak-memory result; the Colab artifact must report measured persistent
state independently of disk cache and peak runtime memory.

## Correctness results

Targeted command:

```text
python -m pytest -q tests/test_tail_fly_math.py tests/test_tail_fly_learner.py tests/test_tail_fly_phasea.py
```

Result: `19 passed`. Covered streaming/batch statistic equality, full-rank
Gram and logit equivalence, diagonal-tail invariants, Woodbury/direct-solve
equivalence, rank-zero behavior, perturbation bound, dynamic class expansion,
deferred batch updates, checkpoint/state audit, deterministic fixed-seed
behavior, strict hidden-test enforcement, cache provenance, and runner resume.

Full repository command:

```text
python -m pytest -q
```

Result: `188 passed`. The complete suite includes all legacy SOHO/FLY and prior
research-branch tests; no existing test regressed.

Notebook/config validation command:

```text
python - <AST and notebook JSON/compile validation script>
Get-FileHash -Algorithm SHA256 configs/tail_fly_imagenetr_train_only.json
```

Result: Python AST and all ten notebook cells compile; locked config SHA-256 is
`c49b6a7e813c94d40413dd2d4f8e5e7889fff9c5b1aea0a5e7af046c0913bc04`.

Warnings are limited to existing PyTorch sparse-CSC beta/invariant notices and
torch JIT deprecation warnings. No correctness test failed because of them.

## File inventory

New method and runner:

- `methods/tail_fly/__init__.py`
- `methods/tail_fly/streaming_svd.py`
- `methods/tail_fly/solver.py`
- `methods/tail_fly/learner.py`
- `tools/tail_fly_phasea.py`
- `configs/tail_fly_imagenetr_train_only.json`

Tests:

- `tests/test_tail_fly_math.py`
- `tests/test_tail_fly_learner.py`
- `tests/test_tail_fly_phasea.py`

Research protocol and execution:

- `docs/research/TAIL_FLY_SPEC.md`
- `docs/research/TAIL_FLY_PHASEA_PROTOCOL.md`
- `docs/research/TAIL_FLY_COLAB_RUNBOOK.md`
- `notebooks/tail_fly_imagenetr_train_only_colab.ipynb`

The branch also inherits the immutable CARS-FLY negative-result record in
`docs/research/CARS_FLY_PHASEA_NEGATIVE_RESULT.md`.

## Remaining gate

Only the user-run ImageNet-R **train-only** Colab development experiment
remains. It may select rank and Ridge from the locked grid, but it must keep
`test.pt` absent. A pass only permits review of a new confirmatory protocol;
a failure stops TAIL-FLY without post-hoc seed, grid, or dataset changes.
