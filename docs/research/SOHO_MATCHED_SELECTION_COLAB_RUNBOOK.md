# SOHO matched-selection Colab runbook

Use `notebooks/soho_matched_selection_final_colab.ipynb` on a Colab GPU. This
is a separate V2 study; it does not modify or resume the V1
`soho_selfcontained_final_colab.ipynb` selection or authorization.

## Comparison contract

All methods use the same frozen ViT-B/16 checkpoint, preprocessing, nested
train-only partitions, class orders, final seeds and test protocol.

- `soho_replay_fidelity` selects density and coding level using the locked
  two-stage SOHO grid. Its per-task Ridge parameter remains the implementation's
  replay-sample GCV policy.
- `flycl_fidelity` is the official/fidelity control fixed at synaptic degree
  300 and coding level 0.3. Its per-task Ridge parameter remains FLY's GCV
  policy.
- `flycl_validation_tuned` selects one of 18 predeclared configurations:
  synaptic degree in `{100,300,500}` and coding level in
  `{0.1,0.2,0.3,0.4,0.45,0.5}`. It uses the same three inner-validation
  replicates as SOHO and raw Ridge. Candidates within 0.05 percentage point of
  the best prefer lower degree, then lower coding level.
- `raw_ridge` selects lambda from the same locked 12-value grid used by V1.

The search spaces are method-specific because their parameter meanings and
feature scales differ. Fairness here means identical data/replicates and a
predeclared, comparable search budget; it does not mean using numerically
identical hyperparameters.

## Cells

1. Edit only repository/path values.
2. Clone the branch, install dependencies and verify protocol/runner hashes.
3. Download the verified ViT checkpoint and three processed datasets.
4. Audit CUB and disclose the legacy ImageNet-R cross-split duplicates.
5. Extract training features only to temporary Colab storage.
6. Run synthetic correctness and fidelity tests.
7. Run train-only nested selection. It is resumable from its output directory.
8. Inspect validation evidence and lock source/config/selection identities.
9. Marks the authorized test boundary.
10. Extract test features only after authorization. The V1 dictionary-loader
    iteration bug is fixed in this V2 runner.
11. Define the final evaluation helper.
12-14. Evaluate CIFAR-100, CUB and legacy processed ImageNet-R separately.
15. Aggregate metrics, paired differences and plots.
16. Export the compact evidence ZIP. Per-sample feature caches and SOHO replay
    tensors are intentionally excluded.

## Important limitations

- Finish an already-authorized V1 run with its existing Colab clone. Do not
  replace its runner or protocol mid-run.
- V2 needs a fresh output and authorization identity. Do not copy V1
  `authorization.json` into the V2 output directory.
- SOHO retains all historical frozen-backbone features and labels and is not
  exemplar-free. The runner counts this sample-level replay state.
- ImageNet-R is a legacy processed split with 19 duplicate content hashes,
  including 18 conflicting-label duplicates. It is not a content-disjoint
  held-out result.
- The repository test splits were consumed by earlier phases. The final stage
  is a locked paired comparative evaluation, not a first-use untouched test.

## Tests

```bash
python -B -m pytest -q \
  tests/test_soho_matched_selection.py \
  tests/test_soho_selfcontained.py \
  tests/test_cached_replay_baselines.py
```

