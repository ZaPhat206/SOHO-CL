# SRQ-FLY state-matched clean-rerun Colab runbook

Run `notebooks/srq_fly_state_matched_final_colab.ipynb` on a Colab T4 GPU.

## Required inputs

Upload both immutable artifacts:

- `srq_fly_p2b_final_confirmation.zip`
- required SHA-256:
  `14826488b8d82bc306a07e6d4f229cc389a8447150833aefc1de664961a9e85d`
- `srq_state_matched_train_only_checkpoint.zip`
- required SHA-256:
  `9c42d3f51581443b642b8b79e793d44f412a73936fc8e45cf9cd7238dcb22801`

This artifact supplies the locked same-width Exact-FLY, P2B and Raw-Ridge
rows. The second artifact supplies the already completed train-only selections.
Both are verified byte-for-byte; no hyperparameter is selected again.

## Cell flow

1. Set paths only.
2. Clone the clean branch and verify runner/config hashes.
3. Verify both input ZIPs.
4. Download the frozen ViT checkpoint and datasets.
5. Audit CUB and the disclosed legacy ImageNet-R split.
6. Extract training features only.
7. Run correctness and leakage tests.
8. Restore the three immutable train-only `selection.json` files. The runner
   verifies the checkpoint hash, retrieves the historical runner from Git,
   verifies its hash, and requires an identical AST for all selection-relevant
   functions before accepting the selections.
9. Review the three restored widths and lambdas.
10. Lock selections, checkpoint provenance, corrected source, Git commit and
    P2B reference.
11. Cross the test boundary, extract/validate test features, and run one
    six-replicate cell per dataset.
12. Summarize, plot and download
    `srq_fly_state_matched_clean_rerun.zip`.

Do not change widths, grid, seeds, task counts, projection, WTA setting or
lambda after the lock. Do not stop a final dataset cell based on interim
accuracy. A disconnected browser can reconnect to a still-running Colab
runtime; rerunning the same cell restores completed units only when its full
context hash matches.

## Expected progress

Selection restoration must print `TRAIN-ONLY SELECTION RESTORE: PASS`. The
review must report width 4,409 and lambda `1e6` for CIFAR, width 4,518 and
lambda `1e5` for CUB, and width 4,518 and lambda `1e6` for legacy ImageNet-R.
The widths are byte-derived, and the lambdas come from the immutable
train-only checkpoint rather than the previous test results.

Final evaluation prints one task line for every stage and one replicate
completion line. The final bundle deliberately excludes feature caches, WTA
codes and the large P2B reference ZIP because none is deployed learner state.
This rerun repairs source provenance only. Since the test splits were already
consumed, its result must still be described as secondary confirmation rather
than fresh held-out evidence.
