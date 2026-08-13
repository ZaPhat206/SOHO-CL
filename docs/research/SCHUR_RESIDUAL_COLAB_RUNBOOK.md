# Schur Residual SOHO Colab runbook

Use `notebooks/schur_residual_cifar100_colab.ipynb` after its commit is pushed
to `feature/crt-soho`.

The notebook reuses the exact Phase E backbone, preprocessing, class order,
train-validation split, fixed anchor and compatible sufficient-statistic
cache. Phase E selected and locked `lambda_p=0.01`, `lambda_r=0.1`, and
`eta=0.1`; this phase does not reopen those grids. Raw Ridge alone receives
the declared `0.01,0.1,1.0` train-validation grid. Schur and every applicable
control receive the same rank grid. Confusion-family controls receive the same
temperature grid; temperature does not apply to Schur, random, or standard
Fisher.

Run all eight cells in order. The runner prints every candidate and records:

- requested and effective rank by stage;
- validation AA and final validation accuracy;
- persistent learner-state bytes;
- absolute and relative solver residuals;
- retained Schur correction energy;
- affinity entropy/CV for confusion controls;
- final principal angles between the proposal and selected controls;
- matched raw-Ridge validation results;
- Gate 0–3 decisions and explicit held-out authorization status.

Even if every train-only gate passes, stop after downloading
`schur_residual_gate_results.zip` and return `gate_results.json` for review.
The notebook intentionally has no held-out evaluation cell. Do not alter rank,
Ridge, thresholds, anchor size, dtype, or class order based on test metrics.

After review confirms all four gates passed, use
`notebooks/schur_locked_heldout_colab.ipynb` for Phase G. Upload the exact
Phase F JSON/ZIP; do not add or edit model hyperparameters. The notebook runs
authorization tests before the runner's single test-cache opening path. See
`docs/research/PHASEG_LOCKED_HELDOUT_PROTOCOL.md` for the immutable contract.
