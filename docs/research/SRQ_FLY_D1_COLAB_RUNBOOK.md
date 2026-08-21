# SRQ-FLY D1 ImageNet-R Colab runbook

Use `notebooks/srq_fly_imagenetr_d1_colab.ipynb` on a Colab T4 GPU. D1 is a
locked 20-task training-validation drift study. It is not held-out evaluation.

## Inputs reused from D0

- train-only frozen ViT feature cache:
  `imagenetr_train_feature_cache_seed2025`;
- FLY-10000 WTA cache: `tail_fly_imagenetr_wta_cache_seed2025`;
- FLY-4096 WTA cache: `srq_fly_wta_h4096_seed2025`.

All three directories must already exist under `MyDrive/T-SOHO`. They contain
sample-level experiment infrastructure and must not be called learner state or
included in an exemplar-free checkpoint.

## Run instructions

1. Push branch `feature/srq-fly-d1`.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Open `notebooks/srq_fly_imagenetr_d1_colab.ipynb`.
4. Edit only repository or Drive paths in cell 2 when necessary.
5. Run cells 1 through 7 in order.
6. Cache copies show one `COPY` line per file. The main runner shows
   `START`, `TASK`, `DONE`, and `RESUME` lines. Paired task lines include exact
   accuracy, SRQ accuracy, prediction agreement, and relative logit error.
7. Download `srq_fly_imagenetr_d1_train_only.zip` and return it for audit.
8. Stop. Do not add a test cache or evaluate held-out ImageNet-R.

Completed method units are context-bound and resumable from the Drive output
directory. A stale config/code/cache identity fails rather than being silently
reused.

## Locked identity

- config: `configs/srq_fly_imagenetr_d1_train_only.json`;
- config SHA-256: `f61da98c3d59d687ce10a4f9ecd5b2ec56251ad4d712130e55d33577a2685dde`;
- seed: `2025`;
- tasks: all `20` training-validation tasks;
- test use: prohibited;
- hyperparameter search: none.
