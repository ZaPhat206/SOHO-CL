# MARS-SOHO Phase 1 implementation

## Status

Implementation and synthetic/train-only runner are ready. No CIFAR-100, CUB or
ImageNet-R Phase-1 study and no held-out evaluation has been run in this phase.
SRQ is intentionally disabled until reconstruction survives the Phase-1 gate.

## Files and roles

- `methods/mars_soho/statistics.py`: streaming normalized class counts, sums,
  squared sums and exact pooled within-class scatter.
- `methods/mars_soho/geometry.py`: solve-based OLDA/ETF construction, temporal
  gauge alignment and hard-Top-K support certificate.
- `methods/mars_soho/reconstruction.py`: deterministic antithetic spherical
  reconstruction, pooled-correlation shrinkage and fixed-budget allocation.
- `methods/mars_soho/learner.py`: exemplar-free Phase-1 learner and matched
  non-exemplar-free exact replay oracle.
- `tools/mars_soho_phase1.py`: fail-closed nested train-only selection, outer
  validation, candidate resume and gates.
- `configs/mars_soho_phase1_train_only.json`: immutable search space and dataset
  identities.
- `tests/test_mars_soho_*.py`: math, state, checkpoint and tiny runner tests.

Existing SOHO, FLY and SRQ implementations were not modified.

## Locked efficient selection design

The Phase-1 expansion width is `1000`. It is a feasibility gate, not a final
paper width. One exact-replay pass evaluates all four ridge lambdas at every
task, avoiding four complete replay executions. Shared and heterogeneous models
search the same four rank/shrinkage combinations with a fixed budget of 64
pseudo-directions per old class. Support-aware and shuffled allocation inherit
the selected heterogeneous configuration and therefore require no additional
inner search.

Old pseudo-directions are concatenated and contribute to `G` through one
weighted matrix product per task, rather than one dense Gram product per class.
Completed candidate/replicate units are written atomically and restored only
when their complete context hash matches.

## Tests completed locally

Environment at implementation time: Windows, repository Python environment,
PyTorch default deterministic CPU execution for tests.

```text
python -m pytest -q tests/test_mars_soho_math.py tests/test_mars_soho_learner.py tests/test_mars_soho_phase1.py
```

Result: `16 passed in 8.29s` on the final focused rerun.

Coverage includes:

- streaming moments equal batch moments;
- solve-based rotation is orthonormal and deterministic;
- certified support never changes in the synthetic check;
- null-space gauge alignment preserves orthonormality;
- pseudo-directions are deterministic, finite and normalized;
- allocation preserves a fixed budget;
- all four reconstruction controls complete two global-inference tasks;
- first task exactly matches the replay oracle;
- checkpoint round-trip has no sample-level state;
- runner refuses visible `test.pt`, uses nested splits and resumes units.

Regression/full-suite command:

```text
python -m pytest -q
```

Result: `324 passed, 20 warnings in 63.28s`. The warnings are pre-existing
TorchScript deprecations and PyTorch sparse CSC/invariant warnings; no warning
originated from the MARS-SOHO namespace.

Direct entrypoint check:

```text
python tools/mars_soho_phase1.py --help
```

Result: exit code 0. This check caught and fixed the repository-root import
path issue before the Colab notebook was finalized.

The worktree also contained pre-existing user edits to
`docs/research/SOHO_MATCHED_TEST_ONLY_RUNBOOK.md` and
`notebooks/soho_matched_test_only_colab_kaggle.ipynb`. Phase 1 did not edit or
stage those files.

## Exact Phase-1 command template

After a train-only feature cache exists and contains no `test.pt`:

```text
python -u tools/mars_soho_phase1.py \
  --config configs/mars_soho_phase1_train_only.json \
  --dataset-key cifar100 \
  --feature-cache-dir <TRAIN_ONLY_CACHE>/cifar100 \
  --output-root <OUTPUT_ROOT> \
  --device cuda
```

Run one dataset per session. Return the dataset directory containing
`phase1_results.json` plus `inner/` and `outer/` evidence. Do not expose or
extract test features.

## Interpretation rule

A failed gate is a research result. Do not widen the grid after seeing outer
validation. Diagnose model error, boundary-risk calibration and oracle gap from
the committed diagnostics first. Do not add SRQ to a failing reconstruction,
because that would confound replay and quantization errors.
