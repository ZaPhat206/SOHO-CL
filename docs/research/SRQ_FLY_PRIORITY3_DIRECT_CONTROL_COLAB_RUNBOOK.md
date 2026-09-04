# SRQ-FLY Priority 3 Colab runbook

Use `notebooks/srq_fly_priority3_direct_control_colab.ipynb` on a T4 GPU. This
is a CIFAR-100 **train-only** structural ablation. It does not evaluate or
authorize the test split.

## What it runs

1. clones the locked experiment branch and checks source hashes;
2. runs synthetic certificate, checkpoint, state, and runner tests;
3. restores or extracts frozen CIFAR-100 training features only;
4. restores or creates one width-10,000 WTA cache;
5. runs Exact FLY, naive direct INT8 Gram, Weyl-repaired direct INT8 Gram,
   FP16 square-root, and current P2B SRQ in isolated processes;
6. plots validation accuracy, persistent state, CUDA peak allocation, and the
   repaired control's diagonal loading;
7. exports `srq_fly_priority3_direct_control_train_only.zip` without feature or
   WTA caches.

## How to run

1. Choose **Runtime -> Change runtime type -> T4 GPU**.
2. Open the notebook and edit path/source values in the first code cell only
   if necessary.
3. Run cells from top to bottom. Do not edit seed, representation, Ridge
   lambda, repair rule, thresholds, or hashes.
4. The expensive cell is resumable by method. `RESUME` means an output was
   source/config verified; `TASK` shows per-task progress.
5. Download and return the final ZIP for audit. Stop there.

The default feature and WTA paths match the earlier Priority-1 notebook so a
live Colab runtime can reuse those caches. They are experiment infrastructure,
not learner state, and are deliberately excluded from the exported artifact.

On a T4 with an existing feature/WTA cache, budget roughly 10-25 minutes for
the five isolated controls. Building the WTA cache or extracting ViT features
can add several minutes. The certified direct control may be the slowest row
because it reconstructs a dense Gram, certifies its quantization error, and
factorizes the repaired system at every task.
