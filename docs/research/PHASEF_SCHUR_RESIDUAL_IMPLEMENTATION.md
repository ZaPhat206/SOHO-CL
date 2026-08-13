# Phase F — Schur Residual SOHO implementation

Status: mathematical/integration gate **PASS**; CIFAR-100 train-validation
experiment **NOT RUN**; held-out test remains unauthorized.

## Implemented result

The explicit method `crt_schur_residual` now derives a low-rank correction
from the Schur complement of the full anchor/raw-residual block Ridge system.
It uses Cholesky, triangular solves, SVD, and Euclidean QR; it never computes an
explicit inverse. The learner's sufficient statistics, exemplar-free
checkpoint inventory, and task-free global inference signature are unchanged.

The optimized gate runner now additionally provides:

- matched raw-Ridge selection on the identical train-validation split;
- generic `--proposal-method confusion_residual|schur_residual` dispatch;
- requested rank and effective rank by task/final stage;
- retained Schur correction energy;
- independently selected controls over the same applicable grids;
- affinity entropy/CV for confusion-family controls;
- final principal-angle diagnostics between proposal/control subspaces;
- generic proposal-versus-strongest-control Gate 3;
- the existing SHA-256 cache audit, numerical Gate 0, early stopping, and
  `test.pt` non-access guarantee.

Phase E Ridge values are locked in the new notebook: anchor `0.01`, residual
`0.1`, complement `0.1`. Only raw Ridge and the new rank hypothesis receive
their declared train-validation grids.

## Mathematical tests

Synthetic tests use fixed seeds and `torch.float64`. They verify:

1. Schur directions are deterministic and Euclidean orthonormal.
2. Retained correction energy is monotone in rank.
3. Rank-one Schur selection attains at least the correction energy of 50 fixed
   random rank-one subspaces.
4. Retaining every supervised correction direction reproduces full
   raw-residual logits with `atol=rtol=1e-9`.
5. Streaming/reconstructed residual statistics and direct block oracles retain
   their earlier `1e-10` or tighter tolerances.
6. `schur_residual` participates in all global-logit, finite-value,
   no-Task-ID, checkpoint, and exemplar-free parameterized tests.

The full-rank equivalence is conditional on fixed positive Ridge/complement
coefficients and retaining every nonzero correction singular direction. The
reduced-rank optimality is for the empirical block Ridge objective represented
by the retained sufficient statistics; it is not a held-out generalization
claim.

## Exact commands and results

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/test_crt_soho_math.py tests/test_crt_gate_runner.py tests/test_experiment_runner.py -q
```

Result: `32 passed`, exit code `0`, pytest runtime `5.60s`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `61 passed`, exit code `0`, pytest runtime `6.03s`.

Warnings: 18 existing `torch.jit` deprecations, one existing sparse-CSC beta
warning, and one sparse-invariant warning while restoring the synthetic gate
cache. No warning was suppressed and no test failed.

`notebooks/schur_residual_cifar100_colab.ipynb` parses as notebook schema 4.5
with eight cells. It has no held-out evaluation cell.

## Required next gate

Push this phase and run only the new train-validation notebook. Return
`gate_results.json` and stop. Schur Residual SOHO is rejected on this protocol
if it fails numerical stability, is more than `0.50` points behind full
residual, or does not exceed the strongest independently selected low-rank
control by at least `0.10` points. Raw Ridge must be reported regardless.

No paper-level claim is supported until this gate passes and later multi-seed,
matched FLY/SOHO, backbone, and dataset evaluation is approved.
