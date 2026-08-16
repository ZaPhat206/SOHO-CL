# TWA-FLY train-only Colab runbook

Use `notebooks/twa_fly_phasea_colab.ipynb` with a T4 GPU. This notebook runs the
locked CIFAR-100 training-validation pilot only. It does not evaluate test data.

1. Upload/open the notebook in Colab and choose **Runtime > Change runtime type
   > T4 GPU**.
2. Run Cell 2 and check the Drive paths. The defaults reuse
   `MyDrive/T-SOHO/tsoho_cifar100_cache` and
   `MyDrive/T-SOHO/zi_soho_wta_h10000_seed1993` from earlier phases.
3. Run Cells 3-6 in order. Cell 4 prints byte progress while restoring large
   caches. Cell 5 extracts features only if the Drive feature cache is absent.
4. Cell 6 must end with `TWA-FLY correctness gate: PASS`.
5. Run Cell 7. It prints one `TASK i/10` line after all analytic candidates for
   that stage finish. A silent interval means a 10000-dimensional Cholesky solve
   or alternating solve is active; do not interrupt unless Colab reports an
   error.
6. Run Cell 8 to inspect the compact table and download the evidence ZIP.

Do not change `rho_candidates`, Ridge policy, seed, task order, projection, or
gate after seeing validation output. Do not rename `test.locked.pt` during Cell
7. Send back `twa_fly_phasea_train_only.zip` whether the gate passes or fails.

Expected cache distinctions:

- `FEATURE_CACHE_DIR`: sample-level frozen features, experiment infrastructure;
- `WTA_CODE_CACHE_DIR`: sample-level WTA codes, experiment infrastructure;
- learner state: only aggregate matrices/configuration and derived weights;
- output: validation metrics, provenance, and gate decision, never a model
  checkpoint containing cached rows.
