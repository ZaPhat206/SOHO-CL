# SRQ-FLY D2.1 Colab runbook

Use `notebooks/srq_fly_imagenetr_d21_lambda_robustness_colab.ipynb` on a Colab
T4 GPU. This is a nested train-only robustness study, not a test evaluation.

## Required Drive artifacts

- `MyDrive/T-SOHO/imagenetr_train_feature_cache_seed2025/{metadata.json,train.pt}`;
- `MyDrive/T-SOHO/srq_fly_wta_h4518_seed2025/` from D2;
- `MyDrive/T-SOHO/srq_fly_imagenetr_d2_state_match_seed2025/d2_results.json`.

The WTA cache contains sample-level training codes and is experiment
infrastructure, never persistent learner state. The output ZIP excludes it.

## Run

1. Push branch `feature/srq-fly-d21-lambda-robustness`.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Open the D2.1 notebook and edit only repository/Drive paths in cell 2.
4. Run cells 1 through 7 in order.
5. `INNER START/TASK/DONE` lines are lambda selection. `OUTER TASK` lines are
   the one locked outer-validation evaluation.
6. Download `srq_fly_imagenetr_d21_lambda_robustness.zip`, return it for audit,
   and stop.

Do not edit the lambda grid, split fractions, seed, representation, gates, or
artifact hashes. Do not create or expose `test.pt`.

## Locked identity

- config: `configs/srq_fly_imagenetr_d21_lambda_robustness.json`;
- config SHA-256: `3c5b54ffedacf5620c8cd9123acb187f5cbf958023b37aebecf9f00c45f73e96`;
- seed: 2025;
- tasks: 20;
- held-out use: prohibited.
