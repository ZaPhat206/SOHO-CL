# SRQ generalization M1 backend extraction

Date: 2026-09-07

## Outcome

M1 extracts the analytic state machinery from the FLY representation without
changing the locked P2B algorithm. The new package contains:

- `ExactGramBackend`, a dense reference implementation;
- `SquareRootBackend`, the fixed P2B blocked-QR and compressed-factor path;
- representation-agnostic blocked QR;
- exact persistent tensor accounting;
- versioned backend checkpoints and import of version-1 SRQ-FLY checkpoints;
- `FLYAnalyticLearner`, a wrapper that owns FlyHash/WTA while delegating
  additive Ridge state to a backend.

The version-1 `CompressedUpper` implementation remains in its historical
module and is exposed through a generic import surface. This preserves locked
checkpoint behavior and avoids an unneeded serialization migration during M1.

## Compatibility evidence

On deterministic synthetic streams, both INT8 and FP16 generic square-root
backends produced:

- byte-identical compressed diagonal, block payload, and scale tensors;
- tensor-identical cross statistics, counts, and classifier weights;
- identical analytic persistent-state bytes after excluding the FLY
  projection from the legacy learner;
- successful generic checkpoint round trips;
- successful import of legacy P2B checkpoints without prediction drift.

The FLY wrapper reproduced the legacy WTA code, compressed factor, classifier,
prediction, and total persistent bytes while the analytic backend source had
no `FlyHash`, coding-level, or synaptic-degree dependency.

## Validation

Focused M1 plus legacy regression command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_analytic_ridge_backend.py tests/test_srq_fly_optimized.py tests/test_srq_fly_learner.py
```

Result: `59 passed, 1 warning in 36.32s`.

Full repository command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `487 passed, 20 warnings in 152.73s`. Warnings were existing PyTorch
JIT deprecations and sparse CSC/invariant notices; no test failed.

No dataset, feature cache, WTA cache, or test split was opened.

## Gate decision

`PASS_M1_GENERIC_BACKEND_EXTRACTION`.

M2 unquantized-equivalence work is authorized locally. No research-scale or
held-out run is authorized.
