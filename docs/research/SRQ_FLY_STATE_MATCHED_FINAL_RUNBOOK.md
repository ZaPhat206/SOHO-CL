# SRQ-FLY Phase 1 state-matched Colab runbook

Run `notebooks/srq_fly_state_matched_final_colab.ipynb` on a Colab T4 GPU.

## Required input

Upload the immutable prior result:

- `srq_fly_p2b_final_confirmation.zip`
- required SHA-256:
  `14826488b8d82bc306a07e6d4f229cc389a8447150833aefc1de664961a9e85d`

This artifact supplies the locked same-width Exact-FLY, P2B and Raw-Ridge
rows. It is verified byte-for-byte and is not re-run.

## Cell flow

1. Set paths only.
2. Clone the clean branch and verify runner/config hashes.
3. Verify the reference ZIP.
4. Download the frozen ViT checkpoint and datasets.
5. Audit CUB and the disclosed legacy ImageNet-R split.
6. Extract training features only.
7. Run correctness and leakage tests.
8. Run one train-only selection cell per dataset. Each cell resumes completed
   lambda/replicate units from local disk.
9. Review the three selected widths and lambdas, then download the small
   train-only checkpoint ZIP.
10. Lock selections, source, Git commit and P2B reference.
11. Cross the test boundary, extract/validate test features, and run one
    six-replicate cell per dataset.
12. Summarize, plot and download `srq_fly_state_matched_final.zip`.

Do not change widths, grid, seeds, task counts, projection, WTA setting or
lambda after the lock. Do not stop a final dataset cell based on interim
accuracy. A disconnected browser can reconnect to a still-running Colab
runtime; rerunning the same cell restores completed units only when its full
context hash matches.

## Expected progress

Selection prints `START`, `TASK`, `DONE`, and `RESTORED`. The selected summary
must report width 4,409 for CIFAR and 4,518 for both 200-class datasets. These
widths are byte-derived, not expected accuracy winners.

Final evaluation prints one task line for every stage and one replicate
completion line. The final bundle deliberately excludes feature caches, WTA
codes and the large P2B reference ZIP because none is deployed learner state.
