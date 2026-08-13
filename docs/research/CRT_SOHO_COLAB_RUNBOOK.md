# CRT-SOHO Colab train-only gate runbook

Use `notebooks/crt_soho_cifar100_colab.ipynb`. This notebook is deliberately
different from the earlier T-SOHO/SFT notebooks: it runs falsification gates
on a deterministic validation subset of cached **training** features and does
not launch held-out test evaluation.

## What is cached

Three storage categories remain distinct:

1. `FEATURE_CACHE_DIR`: frozen ViT features for the experiment. This is disk
   infrastructure and is not learner state.
2. `CRT_GATE_CACHE_DIR`: fixed anchor validation features plus one cumulative
   sufficient-statistic snapshot per task. It is train-only disk
   infrastructure, has a SHA-256 manifest, may contain sample-indexed
   validation data, and is explicitly forbidden from learner checkpoints.
3. Learner state: fixed sparse anchor projection, `G_pp`, `G_xx`, `H_px`,
   `Q_p`, `Q_x`, counts/class mapping, and current bounded derived tensors.
   It contains no historical sample or feature row.

The gate cache is reusable only when the source `train.pt` hash, backbone
metadata, seed, split, anchor configuration, task count, and statistics dtype
all match. A mismatch or corrupted file fails closed.

## Run order

1. Push branch `feature/crt-soho`.
2. Open the notebook in a Colab GPU runtime.
3. Edit only its first configuration cell.
4. Run cells in order. Existing validated ViT and CRT gate caches are reused.
5. Inspect the final validation table and download
   `crt_soho_gate_results.zip`.
6. Stop and send `gate_results.json` for review. Do not add a test-evaluation
   cell yourself, even if `held_out_test_authorized` is `true`.

## Predeclared decisions

- Gate 1 passes when the best full-raw residual validation AA exceeds the
  selected anchor by at least `MINIMUM_FULL_GAIN` percentage points.
- Gate 2 passes when the selected low-rank confusion residual is within
  `MAXIMUM_LOW_RANK_GAP` points of full raw residual.
- Gate 3 passes when the confusion residual exceeds the strongest locked
  random, standard-Fisher, shuffled-confusion, or no-residualization control
  by at least `MINIMUM_CONFUSION_GAIN` points.
- Numerical Gate 0 requires every candidate's maximum relative linear-system
  residual to remain below `MAXIMUM_RELATIVE_SOLVER_RESIDUAL`.

Anchor Ridge is selected first. Its value is locked for all residual methods.
Full residual then selects residual and complement Ridge. Those values are
locked before selecting structured directions. The proposal, shuffled and
no-residualization controls receive the same rank/temperature grid; random and
standard-Fisher receive the same rank grid because temperature does not apply
to them. Gate 3 compares the best train-validation result for every method,
which is conservative toward the proposed method. Test features never
participate.

The default `ANCHOR_DIM=1024` and float32 statistics are a computational pilot,
not final paper settings. Solver residuals are recorded for every candidate.
If numerical residuals are unacceptable, the phase fails; do not silently
change dtype or regularization after viewing held-out results.
