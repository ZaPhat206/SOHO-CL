# SRQ-FLY D2 state-match Colab runbook

Use `notebooks/srq_fly_imagenetr_d2_state_match_colab.ipynb` on a Colab T4.
D2 runs one exact FLY-4518 control against the locked D1 SRQ result. It is a
train-only falsification study, not held-out evaluation.

## Required Drive artifacts

- `MyDrive/T-SOHO/imagenetr_train_feature_cache_seed2025/{metadata.json,train.pt}`;
- `MyDrive/T-SOHO/srq_fly_imagenetr_d1_seed2025/d1_results.json`.

The notebook creates and saves
`MyDrive/T-SOHO/srq_fly_wta_h4518_seed2025` when it does not already exist.
This WTA cache is sample-level experiment infrastructure, not learner state.

## Run

1. Push branch `feature/srq-fly-d2-state-match`.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Open `notebooks/srq_fly_imagenetr_d2_state_match_colab.ipynb`.
4. Edit only repository/Drive paths in cell 2 if necessary.
5. Run cells 1 through 7 in order.
6. Follow `COPY`, `WTA CACHE`, `START`, `TASK`, `DONE`, and `RESUME` lines.
7. Return `srq_fly_imagenetr_d2_state_match.zip` for audit, then stop.

Do not change `m=4518`: it is derived solely from the 105,166,628-byte SRQ
state, and exact FLY-4518 is only 16,780 bytes smaller. Do not expose `test.pt`.

## Locked identity

- config: `configs/srq_fly_imagenetr_d2_state_match.json`;
- config SHA-256: `e8c630b728f9b5f554fd94e6d450b3db4b2205d0d94a595095fa7ebdddcda197`;
- seed: 2025;
- tasks: 20;
- hyperparameter search: none;
- held-out use: prohibited.
