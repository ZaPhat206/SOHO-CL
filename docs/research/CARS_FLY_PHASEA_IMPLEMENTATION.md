# CARS-FLY Phase A implementation report

Date: 2026-08-19  
Branch: `feature/cars-fly`  
Status: implementation and local correctness gates pass; no ImageNet-R
experiment has been run and held-out evaluation is not authorized.

## Implemented scope

- isolated `cars_fly` learner selected explicitly by configuration;
- fixed FlyHash/Top-K anchor and exact dual-view streaming statistics;
- adaptive conditional Schur directions with energy and tail certificates;
- analytic block Ridge using Cholesky/triangular solves, never an explicit
  inverse;
- deterministic rebuild from a checkpoint containing configuration, fixed
  projection, class IDs/counts, and aggregate sufficient statistics only;
- train-only ImageNet-R selection runner with matched controls and physical
  `test.pt` exclusion;
- locked seed `2025`, config hash, checkpoint hash, split hashes, environment,
  diagnostics, state bytes, and gate decision in the output artifact;
- Colab notebook with visible extraction and candidate/task progress.

Existing SOHO and FLY implementations were not modified. The shared experiment
runner received only an explicit CARS-FLY dispatch and diagnostic fields.

## Correctness coverage

The tests cover streaming versus batch sufficient statistics through the
learner, minimal/capped adaptive rank, zero-rank thresholding, energy equality
to the actual regularized objective reduction, exact full-rank equivalence to
the full raw-residual block predictor, deterministic checkpoint round-trip,
state inventory, task-ID-free global logits, invalid configurations, explicit
method dispatch, held-out cache exclusion, and complete class inventory.

Exact focused command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_cars_fly_math.py tests/test_cars_fly_learner.py tests/test_cars_fly_phasea.py tests/test_experiment_runner.py
```

Result: `33 passed, 19 warnings in 11.54s`.

Exact full-suite command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `169 passed, 20 warnings in 27.03s`.

Warnings are pre-existing PyTorch JIT deprecation notices and sparse CSC beta /
sparse invariant notices. No test failed and no warning indicates a numerical
failure in CARS-FLY.

Notebook syntax and diff checks:

```powershell
python -m json.tool notebooks\cars_fly_imagenetr_train_only_colab.ipynb
git diff --check
```

Both commands passed. Line-ending notices from Git on Windows are informational.

## Claim boundary

These tests validate the implementation identities, not the research
hypothesis. They do not show that CARS-FLY improves accuracy, beats FLY-CL, or
forms a useful Pareto point. Only the locked train-validation pilot can decide
whether the Phase A research gates survive. Even a Phase A pass authorizes only
review of a separate held-out protocol.

The per-sample frozen-feature cache used by the experiment runner is disk
infrastructure and is not learner state. It must not be included in a learner
checkpoint or used to claim that the whole experiment workspace is
sample-free. The learner itself is exemplar-free under the state definition in
`docs/CARS_FLY_SPEC.md`.

