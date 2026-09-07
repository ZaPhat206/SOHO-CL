# SRQ generalization M0 baseline lock

Date: 2026-09-07

## Purpose

M0 freezes the evidence and method identity used by the current SRQ-FLY paper
before the learner is refactored into a generic analytic backend. It changes
no method, hyperparameter, dataset, metric, or reported result.

## Baseline identity

- branch at audit start: `experiment/soho-selfcontained`;
- repository HEAD at audit start:
  `1e4c86b86100c4d7089db57136ac638a63b99144`
  (`Repair state-matched rerun provenance`);
- optimized learner SHA-256:
  `40edac2e2cc88faac549f5c87217f3143d815bf53ecad8a37dfdb22c112691ae`;
- optimized storage SHA-256:
  `9d288a3661985da657371e8581f406825d4a8d5e6e0c63381aacda8484490986`;
- evidence manifest: `paper/EXPERIMENTS_MANIFEST.json`;
- staged generalization protocol:
  `docs/research/SRQ_GENERALIZATION_PROTOCOL.md`.

The worktree already contained unrelated uncommitted SOHO/SRQ-SOHO work and
local audit outputs. These files were preserved and were not interpreted as
part of the frozen SRQ-FLY method. `tmp/` and LaTeX build products are ignored
without deleting them. Every later experiment must run from a clean committed
clone even when local development coexists with unrelated work.

## Evidence classes

- P2B same-width three-dataset confirmation: test-used confirmation without
  an accuracy gate; not fresh first-use held-out evidence.
- State-matched three-dataset comparison: test-used secondary recovery
  evidence with a disclosed runtime compatibility adapter.
- Direct quantization control: CIFAR train-only development evidence.
- Task-frequency control: CIFAR train-only five-replicate evidence.
- Whole-process memory: CIFAR train-only, one isolated worker per method on a
  Tesla T4.

The machine-readable manifest records artifact hashes, source commits,
current repository config hashes, test-use semantics, paper roles, and known
caveats. Artifact hashes identify immutable external ZIPs; repository config
hashes identify the current checked-in copies and must not be substituted for
an artifact's internal manifest identity.

## Validation record

Environment: Python `3.13.5`, PyTorch `2.12.0+cpu`. No dataset, feature cache,
WTA cache, or held-out sample was opened.

Machine-readable evidence/protocol gate:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_generalization_m0.py
```

Result: `4 passed in 0.15s`.

Focused SRQ regression gate:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_fly_math.py tests/test_srq_fly_learner.py tests/test_srq_fly_optimized.py tests/test_srq_fly_priority3_direct_control.py tests/test_srq_fly_priority4_task_frequency.py tests/test_srq_fly_priority5_memory.py
```

Result: `77 passed, 19 warnings in 50.50s`.

Full repository gate:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `478 passed, 20 warnings in 74.66s`. Warnings were existing PyTorch
JIT deprecations and sparse CSC/invariant notices; no test failed.

`git diff --check` passed with line-ending conversion notices only. No Colab
run or held-out feature extraction is authorized by this phase.

## Gate decision

`PASS_M0_BASELINE_LOCK_FOR_LOCAL_REFACTOR`.

This decision authorizes M1 local backend extraction. It does not authorize a
large experiment from the current dirty worktree. M3 and later experiment
workers must use a clean committed clone and their own source-locked configs.
