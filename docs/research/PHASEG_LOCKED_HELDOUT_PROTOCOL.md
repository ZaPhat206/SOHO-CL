# Phase G — locked held-out protocol

Status: train-validation authorization **PASS**; held-out implementation gate
**PASS**; held-out CIFAR-100 run **NOT YET EXECUTED**.

## Locked train-validation result

The returned Phase F artifact reports:

| Method | Validation AA | Final validation | State bytes |
|---|---:|---:|---:|
| full raw residual | 91.948088 | 87.100001 | 20,330,904 |
| Schur residual, rank 64 | 91.867770 | 86.700000 | 15,003,032 |
| Fisher residual, rank 64 | 91.694190 | 85.980000 | 15,003,032 |
| raw Ridge | 91.467834 | 86.060001 | 2,974,096 |
| anchor only | 90.719857 | 85.000000 | 14,518,680 |

The maximum relative solver residual was `5.217113263352096e-06`. Full
residual improved the anchor by `1.2282305303073997` percentage points. Schur
rank 64 was `0.08031767512123622` points behind full residual and
`0.17357961715214287` points above the strongest selected low-rank control,
standard Fisher. Its retained correction energy was `0.8078764662981913`.
All four gates passed and the artifact declared held-out authorization.

This is a single seed/split result. It does not establish statistical
significance or a paper-level claim.

## Immutable authorization contract

`tools/schur_locked_eval.py` requires the exact `gate_results.json` and its
SHA-256. Before opening `test.pt`, it:

1. verifies the artifact byte hash;
2. requires the exact train-only protocol/status and four PASS gates;
3. recomputes Gate 0–3 from the complete candidate inventory;
4. verifies raw, anchor, full, Schur, and every control are their respective
   train-validation optima;
5. verifies the proposal is strict low rank and requested rank equals the
   final effective rank;
6. compares semantic frozen-feature metadata plus exact `train.pt` size and
   SHA-256 with the source recorded by selection (`git_commit` remains
   provenance and is not a feature-semantic field);
7. verifies runtime seed/task/anchor/dtype configuration against the artifact.

Newly generated gate manifests also lock `scatter_epsilon` and
`anchor_batch_size`. The already-returned schema-1 Phase F artifact predates
those two fields; its documented notebook defaults (`1e-4` and `1024`) are the
only accepted compatibility values.

Hash, gate, candidate, source-cache, or runtime mismatch terminates before the
single code path that opens `test.pt`. The CLI exposes no rank, Ridge, method,
temperature, or search-grid override. Candidate configurations are read only
from the authorized artifact.

## Final-training and evaluation semantics

After authorization, the runner recomputes cumulative sufficient statistics
using all 50,000 CIFAR-100 training samples, including the selection
validation subset. It reconstructs the fixed anchor from the locked seed and
configuration. At every task it evaluates all globally seen classes without a
Task-ID.

The one locked evaluation includes:

- selected raw Ridge;
- selected anchor only;
- selected full raw residual;
- selected Schur proposal;
- the independently selected random, Fisher, confusion, shuffled-confusion,
  and no-residualization controls.

It records the accuracy matrix, accuracy after each task, final accuracy,
average incremental accuracy, classifier solve/recompute time, end-to-end
inference time (including fixed-anchor encoding), shared sufficient-statistic
streaming time, peak runtime memory, feature-cache disk bytes, persistent state
bytes, effective rank, solver residual, exact class order, locked artifact and
environment. Classifier recomputation timing is reported separately from the
one shared full-stream sufficient-statistic construction and is not presented
as end-to-end online update latency. The output bundle contains an exact byte
copy of the authorized gate artifact, its SHA-256 manifest and the held-out
result.

## Required tests

The test suite proves that SHA mismatch and failed gates occur before any
feature cache access, runtime mismatch occurs without opening test data, and an
authorized toy run uses the locked method inventory and all—not only
selection-training—samples. The held-out notebook must run these tests before
evaluation.

Even a positive single held-out run is not sufficient for a paper claim. The
next gate would require multiple predeclared class-order seeds and paired
uncertainty without further hyperparameter tuning.

## Exact local verification

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/test_crt_gate_runner.py tests/test_schur_locked_eval.py -q
```

Result: `11 passed`, exit code `0`, pytest runtime `19.57s`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `68 passed`, exit code `0`, pytest runtime `13.76s`.

Warnings were 18 PyTorch JIT deprecations, one sparse-CSC beta warning and one
sparse-invariant warning. No warning was suppressed and no test failed. The
notebook JSON check reported eight cells under schema 4.5 with no saved code
cell execution state. No CIFAR-100 held-out evaluation was run locally.
