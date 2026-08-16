# TWA-FLY Phase A implementation

## Decision

TWA-FLY is implemented as a falsifiable train-only pilot, not yet as a claimed
improvement. The held-out CIFAR-100 test cache is not authorized for this phase.

## Why this adaptation of BiCyc

BiCyc addresses temporal representation drift by learning old-to-new and
new-to-old maps around a trainable feature extractor. FLY uses a frozen ViT and
a fixed sparse projection/Top-K transformation, so its representation does not
drift across tasks. Adding temporal maps would therefore manufacture a problem
that is absent and would not transport the sample-dependent Top-K operation.

The applicable principle is paired two-way consistency. TWA-FLY treats raw ViT
features and fixed WTA features as two views, fits one analytic classifier in
each view, and penalizes disagreement of their logits. Its predictor remains
the WTA classifier. The complete formulation, tensor shapes, invariants, and
failure criteria are in `docs/TWA_FLY_SPEC.md`.

## Novelty audit

The agreement objective belongs to the established family of co-regularized
multi-view least squares. Recent analytic continual-learning work also already
covers frozen pretrained features, random expansions, and recursive Ridge. The
research question is narrower:

> Can exact paired raw/WTA cross-statistics provide useful mutual correction
> for FLY's nonlinear Top-K representation, while preserving FLY at rho=0,
> WTA-only inference, and exemplar-free streaming state?

This question is not answered by BiCyc's drift analysis, FLY's single-view
Ridge, or generic multi-view learning in isolation. A paper claim is still
conditional on positive controlled results. Primary sources checked before
implementation:

- Fly-CL (ICLR 2026): fixed sparse expansion, Top-K, streaming Ridge.
- BiCyc (ICLR 2026): bidirectional drift maps and cycle consistency.
- F-OAL (NeurIPS 2024), GACL (NeurIPS 2024), RanPAC (NeurIPS 2023), and
  AnaCP (NeurIPS 2025): analytic/exemplar-free continual learning boundaries.
- Minh et al. (2014): general co-regularized multi-view learning framework.
- Zhao et al. (ICML 2024): statistical role of regularization in continual
  linear regression.

## Files and isolation

- `methods/twa_fly/`: independent statistics, solver, and learner.
- `tools/twa_fly_pilot.py`: train-only runner with no held-out evaluation mode.
- `configs/twa_fly_cifar100_train_only.json`: locked Phase A configuration.
- `tests/test_twa_fly_*.py`: synthetic math, learner-state, and runner tests.
- `notebooks/twa_fly_phasea_colab.ipynb`: Colab execution entry point.

No existing SOHO or FLY implementation is changed.

## Controls

- `matched_fly`: identical projection, WTA codes, global output width, current
  task GCV lambda, and Ridge solve; this is the rho=0 identity.
- `raw_ridge`: same frozen features and train split, locked lambda.
- `twa_one_way`: independent raw teacher distills into WTA Ridge.
- `twa_symmetric`: mutual exact block minimization.
- `twa_shuffled_cross`: destroys raw/WTA sample correspondence within each
  task update while preserving both marginal Grams and supervised targets.

The shuffled control uses `X^T PZ` for a row permutation `P`. Unlike an
arbitrary column shuffle of the cross matrix, this remains a realizable paired
cross-Gram and preserves positive semidefiniteness of the agreement term. This
distinction was caught by the synthetic solver tests before the Colab phase.

## Metrics and state accounting

Validation average incremental accuracy is the arithmetic mean of the ten
per-stage accuracies. Each stage accuracy is evaluated over all validation
samples from classes seen up to that stage. It is not the average over every
entry in the triangular task-by-task matrix.

The runner reports learner-state bytes separately from the WTA experiment cache.
TWA state contains the fixed projection, paired aggregate statistics, counts,
and derived classifiers. The on-disk sample-level WTA code cache is marked
`experiment_cache_not_learner_state` and cannot appear in a learner checkpoint.

## Exact local commands

Targeted Phase A tests:

```text
python -m pytest -q tests/test_twa_fly_math.py tests/test_twa_fly_learner.py tests/test_twa_fly_pilot.py
```

Full suite (must be run immediately before handoff):

```text
python -m pytest -q
```

Observed locally on Python 3.13.5 and torch 2.12.0+cpu:

- targeted command: `14 passed, 19 warnings in 4.53s`;
- full command: `145 passed, 20 warnings in 17.95s`.

The warnings are existing torch JIT deprecations and sparse CSC/invariant beta
warnings. No TWA-FLY test failed, and the full suite introduced no regression.

## Phase A stopping rule

The notebook physically renames local `test.pt` before selection. It restores
the local filename afterward, but the runner itself has no test-evaluation
path. A PASS only requests review for a separately specified held-out phase.
Any failed gate ends this direction without test-time tuning.
