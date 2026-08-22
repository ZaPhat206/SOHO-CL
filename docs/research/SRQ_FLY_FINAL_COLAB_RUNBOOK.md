# SRQ-FLY final three-dataset Colab runbook

Use `notebooks/srq_fly_final_three_dataset_colab.ipynb` with a T4 GPU. This is
the first notebook in the project that is authorized to materialize held-out
features under the locked manifest.

## Files required from the user

When cell 6 opens the upload widget, select these four small evidence ZIPs:

1. `srq_fly_cifar_d5_train_only.zip`;
2. `srq_fly_cub_d4_multiseed_train_only.zip`;
3. `srq_fly_imagenetr_d21_lambda_robustness.zip`;
4. `srq_fly_imagenetr_d1_train_only.zip`.

The notebook downloads the datasets and verified ViT checkpoint itself. It
uses local `/content` disk and does not require Google Drive.

## Cell order

1. Read the scope and disclosure.
2. Edit repository/path values only. Do not edit hashes, seeds or method
   settings.
3. Clone/install and verify a clean immutable repository.
4. Download the checkpoint and three Kaggle datasets.
5. Audit CUB and the disclosed legacy ImageNet-R split.
6. Verify completed train-only selection artifacts. This is the train-only
   hyperparameter gate; test data remains unopened.
7. Extract train features only. Progress is one line per task.
8. Run synthetic correctness and leakage tests.
9. Read the single-use boundary.
10. Write the immutable authorization record.
11. Extract held-out test features. Progress is one line per task.
12. Define the resumable dataset helper.
13. Run CIFAR-100: six seeds and four methods.
14. Run CUB-200-2011: six seeds and four methods.
15. Run ImageNet-R legacy processed split: six seeds and four methods.
16. Aggregate `mean ± std`, confidence intervals and download
    `srq_fly_three_dataset_heldout_results.zip`.

During evaluation:

- `WTA CACHE` lines are one-time sample-level experiment-cache progress;
- `UNIT START/RESTORED/DONE` identifies resumable method units;
- `TASK` reports the current stage, accuracy and paired agreement;
- `SEED DONE` and `DATASET COMPLETE` are the safe boundaries.

Do not delete feature/WTA caches while a run is active. If Colab disconnects
but the runtime survives, rerun the interrupted dataset cell. A fresh runtime
loses `/content`; because Drive storage was deliberately avoided, train/test
features and WTA codes must then be regenerated.

The downloaded evidence ZIP excludes feature/WTA caches and therefore does
not contain sample-level data. Return that ZIP for audit before editing the
paper's results tables.
