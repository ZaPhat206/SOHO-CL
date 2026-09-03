# SRQ-FLY P2B final confirmation runbook

Use `notebooks/srq_fly_p2b_final_confirmation_colab.ipynb` on a Colab T4 GPU.
This is the final confirmation of the selected Priority-2B runtime backend. It
does not tune a method and it does not authorize the rejected Priority-2C
implicit-Ridge initialization.

## Scientific status

The notebook reuses the immutable train-only choices from
`srq_fly_selfcontained_three_dataset_results.zip`:

| Dataset | FLY/SRQ Ridge lambda | Raw-Ridge lambda | Tasks |
|---|---:|---:|---:|
| CIFAR-100 | 1,000,000 | 100 | 10 |
| CUB-200-2011 | 100,000 | 100 | 20 |
| ImageNet-R | 1,000,000 | 1,000 | 20 |

It runs the same six final replicate pairs from the base protocol. Exact FLY
and P2B share each projection, WTA cache, class order, task split, backbone and
Ridge value. Raw Ridge shares the backbone, class order and task split and uses
its separately selected Ridge value.

The three test splits were already consumed by the earlier locked SRQ-FLY
implementation. The output must therefore be described as an
**optimized-backend confirmation**, not as a new first-use held-out result.
ImageNet-R is additionally a legacy processed split with 19 cross-split
duplicate content hashes, 18 under conflicting labels; it is not
content-disjoint.

## Required input

Before cell 4, upload this exact file through the Colab Files sidebar:

`srq_fly_selfcontained_three_dataset_results.zip`

Expected SHA-256:

`e4b630781ff6f69deaecb63dda9926d256cd6b654ef4b51a682bf3ef94e6490b`

The notebook extracts only the three selection JSON files. It does not trust
or reuse prior test metrics.

## Cell flow

1. States the study purpose and the prior-test-use disclosure.
2. Declares editable repository and scratch paths. Method settings are locked.
3. Fresh-clones the branch, installs dependencies, checks GPU and source hashes.
4. Restores and cryptographically verifies the three train-only selections.
5. Downloads the frozen ViT-B/16 checkpoint and processed datasets.
6. Audits CUB and ImageNet-R dataset identities without extracting features.
7. Extracts training features only; `test.pt` remains absent.
8. Runs correctness, backend and test-boundary checks.
9. Creates the immutable authorization with the unchanged base protocol.
10. Marks the test boundary.
11. Extracts test features only after authorization.
12. Defines the resumable per-dataset command.
13. Runs CIFAR-100: six paired replicates and three methods.
14. Runs CUB: six paired replicates and three methods.
15. Runs legacy ImageNet-R: six paired replicates and three methods.
16. Computes mean, sample standard deviation, paired 95% intervals and tables.
17. Draws AIA, final accuracy, persistent state and normalized task curves.
18. Exports the compact evidence ZIP and excludes all feature/WTA caches.

## Method identity

The accepted P2B backend is fixed to:

- int8 groupwise square-root storage;
- blocked QR rank update with panel size 128;
- Gram-Cholesky initialization for the first task;
- streaming factor quantization in batches of 64 blocks.

Priority-2C's implicit-Ridge first update is not used because it failed the
pre-registered predictor-equivalence gate on real CIFAR-100 training data.

## Running and recovery

Run cells from top to bottom on a T4. Each completed dataset/replicate is saved
under `/content/srq_p2b_confirmation_results`; rerunning a failed dataset cell
restores matching completed units. Local `/content` disappears when the Colab
runtime is destroyed, so do not deliberately terminate the runtime mid-study.
If desired, change only the scratch root paths in cell 2 to a mounted persistent
volume before starting; never change config, seed, lambda or backend values.

At completion, return `srq_fly_p2b_final_confirmation.zip` for audit. The ZIP
contains metrics, task curves, authorization, source/config snapshots,
selection evidence and plots. It deliberately excludes sample-level frozen
features and WTA codes because those are experiment infrastructure rather than
persistent learner state.
