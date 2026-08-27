# WBT-SOHO Phase 1D CIFAR-100 Colab runbook

Use `notebooks/wbt_soho_phase1d_cifar100_colab.ipynb` on a Colab GPU. This is
a nested **train-only** feasibility study. It is not held-out evaluation and
must not be reported as test accuracy.

## What it does

1. clones the locked branch/commit and verifies repository cleanliness;
2. runs synthetic mathematics, learner-state and runner tests;
3. restores only `metadata.json` and `train.pt` from the existing frozen ViT
   feature cache, or uploads a train-only cache archive;
4. physically checks that `test.pt` is absent;
5. runs the exact oracle once per split/seed and caches only reference `G,Q`;
6. selects four boundary configurations on inner training validation;
7. evaluates the six locked controls on outer training validation;
8. prints a compact table, gate decision and downloads an evidence ZIP.

The notebook prints `START`, one `TASK` line per incremental stage, and `DONE`
for every resumable unit. Completed units under `OUTPUT_ROOT` are restored
after interruption when their context hashes match.

## Locked protocol

- dataset: CIFAR-100 training split only, 10 class-incremental tasks;
- frozen feature dimension: 768;
- SOHO expansion dimension: 1000;
- density/coding level: 0.1/0.4 with ETF enabled;
- tangent rank: 16;
- Ridge lambda: 10;
- pseudo rows per old class: 64;
- boundary fraction: `{0.25, 0.50}`;
- boundary strength: `{0.25, 0.50}`;
- three development replicates from the immutable config;
- seed policy and class order come from the config, never notebook edits.

## How to run

1. Select **Runtime -> Change runtime type -> T4 GPU**.
2. Edit only repository branch/path and train-cache source values in cell 2.
3. Run cells in order.
4. If the cache archive is uploaded, it must contain only `metadata.json` and
   `train.pt`. Do not upload or restore `test.pt`.
5. Wait for `PHASE 1D PROCESS: COMPLETE`, then run the result/export cell.
6. Download `wbt_soho_phase1d_cifar100_train_only.zip` and return it for audit.
7. Stop. Even a pass only authorizes review of the next phase.

The frozen feature cache contains per-sample training embeddings and is
experiment infrastructure on disk. It must not be shipped in a learner
checkpoint or counted as persistent exemplar-free learner state.
