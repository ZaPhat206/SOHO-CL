# TWA-FLY D0 Colab runbook

Use `notebooks/twa_fly_d0_colab.ipynb` with a T4 GPU. D0 reuses the existing
CIFAR-100 feature and H=10000 WTA caches and never evaluates test features.

1. Open the notebook and select **Runtime > Change runtime type > T4 GPU**.
2. In Cell 2, edit paths only if your Drive layout differs. Do not edit seed,
   Ridge coefficients, fusion alphas, class order, or gate thresholds.
3. Run Cells 3–5. Cell 4 displays byte progress while restoring caches. Cell 5
   must say `extraction skipped` when the existing feature cache is present.
4. Cell 6 must finish with `TWA-FLY D0 correctness gate: PASS`.
5. Run Cell 7. It first verifies/upgrades projection provenance, then prints one
   compact line per task containing FLY/raw/oracle accuracy, raw-only correct
   count, disagreement, task time, and total time.
6. Run Cell 8, download `twa_fly_d0_train_only.zip`, and return that ZIP whether
   the gate passes or fails.

The runner temporarily renames `test.pt` to `test.locked.pt` and restores the
filename in a `finally` block. The runner itself fails if `test.pt` is visible.
The cache may contain sample-level rows because it is experiment infrastructure;
no learner checkpoint is created and no method result is called exemplar-free
based on the cache.

Expected decision meanings:

- `REVIEW_JOINT_RESIDUAL_IMPLEMENTATION`: raw errors are sufficiently
  complementary to justify implementing a new analytic joint scorer. Stop and
  return the ZIP for review; do not evaluate test.
- `STOP_TWO_VIEW_BRANCH`: the raw view has insufficient usable headroom. Do not
  search another seed or dataset from this notebook.

