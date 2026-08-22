# SRQ-FLY self-contained final Colab runbook

Use `notebooks/srq_fly_selfcontained_final_colab.ipynb` on a Colab T4 GPU.
It downloads the verified checkpoint and all three datasets, and it stores
temporary feature/WTA caches on `/content`; Google Drive is not required.

## What is automated

The notebook performs, in order:

1. clone the committed branch and verify immutable protocol/runner hashes;
2. download and verify the frozen ViT checkpoint;
3. download CIFAR-100, CUB-200-2011, and ImageNet-R from Kaggle;
4. audit CUB and the disclosed legacy ImageNet-R processed split;
5. extract training features only while `test.pt` remains absent;
6. run the mathematical, state, split-leakage, and notebook tests;
7. run the locked 12-value train-only Ridge grid on three development
   replicates per dataset;
8. display the chosen values and outer-validation evidence, then write the
   immutable authorization;
9. extract official test features only after authorization;
10. refit the analytic learners on all training features and evaluate six
    fresh replicates for each dataset;
11. aggregate tables, create six report figures, and download a compact ZIP.

## How to run

1. In Colab select **Runtime → Change runtime type → T4 GPU**.
2. Open the notebook and edit only cell 2 if the repository URL or branch has
   changed. Do not edit hashes, seeds, grid, method settings, or gates.
3. Run cells from top to bottom.
4. During feature extraction, wait for one progress line per task.
5. During selection, `START`, `RESTORED`, and `DONE` identify resumable
   candidate units. The complete selection is 12 lambdas × 3 development
   replicates × three datasets, with paired Exact/SRQ work sharing codes.
6. If a selected lambda is a grid endpoint, stop and return the selection
   files. Do not run the authorization or test cells.
7. After successful locking, run the three final dataset cells and then the
   aggregation, plots, and export cells.
8. Download `srq_fly_selfcontained_three_dataset_results.zip` before the
   Colab runtime ends and return it for audit.

Because caches live only on `/content`, a new Colab runtime must regenerate
them. A surviving runtime can safely resume completed selection/evaluation
units. Do not manually copy or edit a selection, authorization, result, or
cache file.

## Figures produced

- `01_accuracy_by_task.png`: mean seen-class accuracy versus task fraction;
- `02_final_and_aia.png`: final accuracy and average incremental accuracy;
- `03_persistent_state.png`: persistent learner-state MiB;
- `04_accuracy_memory_pareto.png`: AIA versus persistent state;
- `05_forgetting.png`: forgetting by dataset and method;
- `06_paired_seed_differences.png`: paired SRQ minus Exact-FLY AIA per final
  replicate.

Error bars use the six final replicates. Runtime plots are deliberately not a
paper comparison unless all methods share an identical environment and timing
scope. The ZIP includes results, selections, authorization, plots, config, and
source provenance, but excludes sample-level feature and WTA caches.
