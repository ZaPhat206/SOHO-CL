# Phase I — preregistered CUB train-only selection

Status: **implementation complete; experiment not run**. The held-out CUB test
split is not authorized for model evaluation.

## Research question

Phase H found a small, consistent CIFAR-100 average-incremental advantage for
rank-64 Schur residual over raw Ridge, but no established final-accuracy gain.
Phase I asks whether that effect transfers to a fine-grained dataset and
whether it survives a fair, method-specific train-only search. CUB was chosen
before any CUB model score was observed.

This phase has two explicitly separated arms:

1. **Fixed transfer:** reuse the complete locked CIFAR-100 configuration
   unchanged. This measures transferability rather than optimality.
2. **Equal-budget train-only selection:** independently select Schur, Fisher,
   and random residual methods from the same `3 ranks × 3 residual Ridge × 3
   complement Ridge = 27` candidates. Raw Ridge and anchor-only each receive
   four Ridge candidates. Full residual receives the declared Cartesian grid.

The fixed-transfer and selected scores are validation results only. Neither is
a CUB test result.

## Locked data and representation

- Dataset source used in Colab: Kaggle
  `zaphat206/cub-200-2011`.
- Processed layout: `ImageFolder/train` and `ImageFolder/test` with identical
  200-class mapping.
- Train images: `5,994`; held-out test images: `5,794`.
- Dataset identity SHA-256:
  `e374af9b576cb6b3503198ef3ea30fd0aa9d2e18c230ff8064e21d4f644af2ca`.
- Cross-split duplicate image contents: zero.
- Frozen backbone: `vit_base_patch16_224` loaded from
  `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`.
- Checkpoint SHA-256:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`.
- Preprocessing: resize 256, center crop 224, tensor conversion, normalize by
  mean/std `(0.5, 0.5, 0.5)`; no train augmentation.
- Protocol: 20 tasks, 10 new classes per task, deterministic class order from
  seed `1993`, stratified `20%` train validation.

The notebook extracts **train features only**. It records the expected test
count from the directory inventory but never opens a test image and never
creates `test.pt`. The selection runner calls `validate_cache(...,
load_test=False)` exclusively.

## Locked grids and gate

The immutable manifest is
`configs/phasei_cub_train_only_selection.json`. Version 2 uses `float64`
sufficient statistics and analytic solves. Version 1 used `float32`; its first
`λ=0.001` raw-Ridge candidate failed Cholesky before producing any validation
score, so it is invalidated as a numerical preflight failure rather than used
to alter the search grid. Current manifest SHA-256:
`e234b97080442c113578c7d477a2eecfc60ad2a9484fbea42ea1720fd9dd62d9`.
Its exact grids are:

- raw/anchor Ridge: `0.001, 0.01, 0.1, 1.0`;
- residual/complement Ridge: `0.01, 0.1, 1.0`;
- low ranks: `32, 64, 128`;
- low-rank controls: Schur, standard Fisher, and seeded random;
- selection metric: validation average incremental accuracy;
- exact ties: first value in preregistered order.

The held-out CUB test can be proposed for a later, separately reviewed phase
only if all of these train-only checks pass:

- maximum relative solver residual ≤ `1e-4`;
- selected full residual exceeds selected anchor-only by ≥ `0.1` point;
- selected Schur is within `0.5` point of selected full residual;
- selected Schur exceeds selected raw Ridge by ≥ `0.1` point;
- selected Schur exceeds the stronger independently selected Fisher/random
  low-rank control by ≥ `0.1` point.

Confusion-based methods are excluded because matched and shuffled controls
already falsified their claimed geometric benefit in Phases B and E. FLY and
replay SOHO do not select CRT parameters and will only enter a later locked
held-out comparison.

## Learner-state and cache distinction

- Raw CUB images and frozen train features are experiment infrastructure.
- The reusable gate cache contains validation indices/features and cumulative
  sufficient-statistic snapshots. It is forbidden in a learner checkpoint.
- Candidate JSON files are resumability infrastructure.
- Exemplar-free learner state remains limited to analytic sufficient
  statistics, fixed projection metadata, class mapping, and fitted analytic
  parameters; it contains no historical per-sample feature or label.

## Commands for this implementation phase

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_cub_dataset_audit.py tests/test_cub_data_utils.py tests/test_phasei_cub_selection.py tests/test_crt_gate_runner.py
```

Focused result for the command above: `16 passed`. An expanded related suite
including `tests/test_experiment_runner.py` produced `28 passed`. Final
repository command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result after the float64 numerical-preflight amendment: `95 passed` in
`12.89s`. Both commands emitted only the existing
PyTorch JIT deprecations and sparse CSC/invariant warnings; no warning was
suppressed. `git diff --check` passed, the locked manifest hash was recomputed,
and all nine notebook code cells passed Python syntax parsing.

## User gate

Run `notebooks/phasei_cub_train_only_colab.ipynb` in order and return
`phasei_cub_train_only_selection.zip`. Do not create or run a held-out CUB
evaluation cell. A PASS artifact authorizes review, not automatic test access.
