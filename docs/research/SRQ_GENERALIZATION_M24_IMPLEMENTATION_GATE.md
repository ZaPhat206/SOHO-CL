# M24 implementation gate

Date: 2026-09-17

## Scope

This gate covers only the implementation and small deterministic tests for
the packed-Exact and Frequent-Directions controls.  No CIFAR representation,
validation accuracy or test accuracy was computed, and the research-scale
M24 run has not started.

## Record

```text
phase: M24 implementation and algebra/accounting controls
git_commit: pending isolated M24 commit
worktree_status: dirty before M24; unrelated user changes preserved
commands_exact:
  python -B -m pytest -q -p no:cacheprovider tests/test_equal_memory_controls.py tests/test_srq_generalization_m24.py
  python -B -m pytest -q -p no:cacheprovider tests/test_equal_memory_controls.py tests/test_srq_generalization_m24.py tests/test_analytic_ridge_backend.py tests/test_ranpac_analytic_frontend.py
exit_codes: 0, 0
tests_passed/total: 10/10, then 24/24
environment: Windows; Python 3.13; local CPU; PyTorch from repository environment
backbone_and_checkpoint: not loaded; config locks ViT-B/16 SHA-256 32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b
dataset_and_split: not loaded; runner locks the verified M23 CIFAR-100 train-only split identities
class_order_hash: not computed locally; must equal each verified M23 stream at runtime
seed(s): 2025, 2026, 2027 (historical M23 stream identities retained)
config_hash: to be locked in the pinned Colab notebook after commit
state_inventory_and_bytes:
  packed Exact: projection + physical unique FP32 Gram entries + Q + counts + weights; width 5878; 91873540 bytes
  FD-Ridge: full projection + FP32 1328x10000 FD summary + Q + counts + weights; 91840400 bytes
checkpoint_audit: unit round trips pass; no sample-level field is retained
metrics: none
gate_decision: pass for preparing the pinned train-only Colab run
known_deviations: update-workspace peaks are measured but are not part of the persistent-state budget; FD uses repeated exact SVDs and may be slow
```

The large run remains blocked until the notebook pins an immutable commit,
verifies the exact M23 artifact, passes the focused tests, creates only
`train.pt`, and confirms that `test.pt` is absent.
