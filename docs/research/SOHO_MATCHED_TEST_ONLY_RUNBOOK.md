# SOHO matched-selection locked test-only runbook

Use `notebooks/soho_matched_test_only_colab_kaggle.ipynb` only after the
three-dataset train-only selection has completed. The notebook does not search
or alter hyperparameters.

## Locked configurations

| Dataset | SOHO density / coding | Tuned FLY degree / coding | Raw Ridge lambda |
|---|---|---|---:|
| CIFAR-100 | 0.1 / 0.4 | 500 / 0.4 | 100 |
| CUB-200-2011 | 0.5 / 0.45 | 500 / 0.2 | 100 |
| ImageNet-R | 0.2 / 0.45 | 300 / 0.5 | 1000 |

These values are preregistered in
`configs/soho_matched_selected_hyperparameters.json`. The notebook also
requires the original `selection.json` for every dataset. It refuses to create
test features when the files are missing, report `uses_test_set=true`, have a
source hash mismatch, or disagree with the preregistered values.

## Prepare the Kaggle input

The selection-only notebook output should contain either:

- `soho_matched_selection_train_only.zip`; or
- a directory containing `cifar100/selection.json`, `cub200/selection.json`
  and `imagenetr/selection.json`.

Attach that saved Kaggle output as an input to the test notebook. If its ZIP has
a different name, set `SELECTION_ARTIFACT` in cell 2 to its exact path.

## Execution order

1. Edit only repository/path values.
2. Clone and verify protocol, runner, base-runner and locked-manifest hashes.
3. Restore and verify the original three-dataset selection evidence.
4. Download the exact frozen checkpoint and processed datasets.
5. Audit CUB and disclose the legacy ImageNet-R duplicate content.
6. Extract training features only; test remains hidden.
7. Run unit tests and create the immutable authorization.
8. Cross the test boundary and extract test features.
9. Run CIFAR, CUB and ImageNet-R in separate cells. Each performs six paired
   replicates for SOHO replay, official FLY, validation-tuned FLY and raw Ridge.
10. Aggregate metrics and export the evidence ZIP.

Do not change any hyperparameter after the authorization cell. SOHO replay is
not exemplar-free because its learner state retains historical frozen-backbone
features and labels. ImageNet-R remains a legacy processed split and is not a
content-disjoint held-out result.

