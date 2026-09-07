# SRQ generalization M3 FLY regression runbook

Date: 2026-09-07

Status: `READY_FOR_COLAB_M3`; the real CIFAR train-only gate has not yet been
executed.

## Purpose

M3 checks that extracting the analytic state into a generic backend did not
change either FLY path used by the locked paper evidence:

1. the historical Exact-FLY Gram update versus `ExactGramBackend` behind the
   FLY frontend;
2. the historical optimized P2B learner versus `SquareRootBackend` behind the
   same FLY frontend.

This is an implementation-regression experiment, not a new accuracy
comparison and not hyperparameter selection. Both sides receive the same
precomputed WTA rows and labels. The gate requires tensor identity, persistent
byte identity, zero relative logit drift, 100% prediction agreement, and the
locked solver-residual tolerance after every task.

## Locked experiment

- Dataset materialized: CIFAR-100 training split only.
- Frozen backbone: ViT-B/16 with the already locked checkpoint SHA-256.
- Schedule: 10 class-incremental tasks, seed 2025.
- Regression probe: deterministic 80/20 stratified split of training samples.
- FLY: width 10,000, synaptic degree 300, coding level 0.3.
- Ridge: `1e6`, inherited from the existing train-only selection evidence.
- P2B: mixed INT8/FP32, block 256, group 64, blocked-QR panel 128, first
  update by Gram--Cholesky, streaming quantization in batches of 64 blocks.
- Held-out test: never extracted or loaded.

The notebook verifies SHA-256 identities of the configuration, runner,
generic backend, FLY frontend, and legacy learner before downloading data. It
also requires a clean Git checkout. Export excludes the feature and WTA
caches because both contain sample-level material and are experiment
infrastructure rather than learner state.

## Local preflight evidence

Focused command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_generalization_m3.py `
  tests/test_analytic_ridge_backend.py `
  tests/test_analytic_ridge_equivalence.py
```

Result: `16 passed, 19 warnings in 10.95s`.

The synthetic end-to-end test uses six classes across three tasks and proves
exact legacy/generic identity for the Gram, cross-statistic, counts, weights,
compressed P2B factor payload/scales/diagonal, state bytes, logits, and
predictions. Every notebook code cell compiles, and its train-only/source-lock
contract is tested.

Full repository command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `494 passed, 20 warnings in 121.72s`. Warnings were existing PyTorch
JIT deprecations and sparse tensor notices; no test failed.

## Colab procedure

Open and run every cell in:

`notebooks/srq_generalization_m3_fly_regression_colab.ipynb`

Expected terminal status:

```text
PASS_M3_FLY_REGRESSION
```

Return `srq_generalization_m3_fly_regression.zip`. The ZIP must contain only
`m3_results.json` and the locked config. M3 is not marked PASS until that ZIP
is audited against the committed source and all real-stream task gates.

## Effect if M3 passes

A pass will establish that the generic backend is a behavior-preserving
refactor for FLY on the real train-only development stream. It will authorize
M4, where the same backend is attached to a genuinely different expanded
feature frontend (RanPAC). It will not by itself justify a plug-in claim,
because FLY remains the only evaluated frontend until M4 passes.
