# SRQ generalization M3 FLY regression runbook

Date: 2026-09-07

Status: `PASS_M3_FLY_REGRESSION`.

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

## Audited Colab result

Open and run every cell in:

`notebooks/srq_generalization_m3_fly_regression_colab.ipynb`

Artifact: `srq_generalization_m3_fly_regression.zip`.

- ZIP bytes: `2479`;
- ZIP SHA-256:
  `ec376875ff0e29e2d7a46d1fcb9fcccc523ea47fbe6a8f2dd83ec3a98930a595`;
- result SHA-256:
  `bc2851c75362ac4bda8cd0ab022d21c4f952b1310dafbabaa7cf74452b98514c`;
- source commit: `56d525f453744f13ff287507f6ec4c30aaf81644`;
- source checkout: clean;
- test use: false;
- accuracy-based selection: false.

The archive passed CRC validation, contained exactly `m3_results.json` and
`config.json`, had no duplicate or unsafe paths, and its config was
byte-identical to the committed configuration.

All ten task records passed every locked identity check. Historical and
generic Exact FLY had identical Gram, cross-statistic, counts, classifier,
class IDs, persistent bytes, logits, and predictions. Historical and generic
P2B had identical compressed factor diagonal, INT8 payload, scales,
cross-statistic, counts, classifier, class IDs, persistent bytes, logits, and
predictions.

Final audit summary:

```text
minimum prediction agreement       1.0
maximum relative logit error       0.0
maximum solver relative residual   3.132620984020085e-06
Exact FLY final state               444006540 B
P2B final state                     97166228 B
P2B state reduction                 78.11603675927836%
```

## Effect of M3

M3 establishes that the generic backend is a behavior-preserving refactor for
FLY on the real train-only development stream. It authorizes
M4, where the same backend is attached to a genuinely different expanded
feature frontend (RanPAC). It will not by itself justify a plug-in claim,
because FLY remains the only evaluated frontend until M4 passes.

Gate decision: `PASS_M3_FLY_REGRESSION`.
