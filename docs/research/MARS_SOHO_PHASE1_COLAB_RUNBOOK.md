# MARS-SOHO Phase 1 Colab runbook

Use `notebooks/mars_soho_phase1_train_only_colab.ipynb` with a T4 GPU. The
notebook evaluates one dataset per session and never extracts test features.

## Run order

1. Push the implementation commit on `experiment/soho-selfcontained`.
2. Open the notebook and set only `DATASET_KEY` in cell 2:
   `cifar100`, `cub200`, or `imagenetr`.
3. Run all cells from top to bottom.
4. The extraction cell downloads only the chosen dataset and creates
   `train.pt`; it asserts that `test.pt` is absent.
5. The correctness cell must report all MARS-SOHO tests passed.
6. The locked run prints `START`, `RESTORED`, `RIDGE GRID`, `TASK`, and `DONE`
   progress. A long `TASK` line means a weighted Gram or solve is running.
7. Inspect the outer-validation table and gates. These are train-only results,
   not held-out test results.
8. Download `mars_soho_phase1_<dataset>_train_only.zip` and return it for audit.

Run datasets sequentially. Do not alter the grid, seeds or gates after seeing
the first dataset. The output ZIP excludes the feature cache because that cache
contains sample-level embeddings and is experiment infrastructure, not learner
state.

## What a pass means

A per-dataset `phase1_pass` says that, on its untouched outer partition, the
support-aware reconstruction met all three predeclared approximation/control
thresholds. It does not authorize held-out test evaluation or Phase-2 SRQ by
itself. Cross-dataset review is required after all three artifacts are audited.

If a gate fails, return the ZIP unchanged. Do not enlarge the grid or use test
accuracy to repair it.
