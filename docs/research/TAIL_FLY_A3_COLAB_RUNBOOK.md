# TAIL-FLY A3 Colab runbook

Use `notebooks/tail_fly_imagenetr_a3_colab.ipynb` on a Colab GPU runtime and
run its seven cells in order.

The notebook restores the exact Phase A train-only ImageNet-R feature cache
and WTA cache from these default Drive locations:

- `MyDrive/T-SOHO/imagenetr_train_feature_cache_seed2025`
- `MyDrive/T-SOHO/tail_fly_imagenetr_wta_cache_seed2025`

It deliberately fails if either cache is missing. Do not regenerate data or
edit the locked seed/grid to make A3 pass. The output uses a new directory,
`MyDrive/T-SOHO/tail_fly_imagenetr_phasea3_seed2025`, so Phase A evidence is
not overwritten.

Expected progress is concise:

- the restore cell prints one `COPY` line per cache file;
- the runner prints `START`/`DONE` per unit and one `TASK` line per continual
  stage;
- completed units from the same implementation print `RESUME`;
- stale units from another commit or source implementation fail explicitly.

At completion, download `tail_fly_imagenetr_phasea3_train_only.zip` and return
it for audit. Do not expose or evaluate ImageNet-R `test.pt`.
