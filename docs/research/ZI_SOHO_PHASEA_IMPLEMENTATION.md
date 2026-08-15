# ZI-SOHO Phase A implementation

Status: implementation and synthetic correctness gate pass. The CIFAR-100
train-only pilot has not been run, and held-out evaluation is not authorized.

## What was implemented

`methods/zi_soho` adds an opt-in fixed-WTA analytic learner without changing
the original SOHO or FLY implementations. It retains a fixed sparse projection
and only class counts plus per-coordinate active counts, sums and squared sums.
No image, backbone feature, sparse code, label history or Task-ID is retained
by the learner.

The four scorers are `wta_ncm`, `support_only`, `active_gaussian` and the
proposed `hurdle` combination. The implementation evaluates only the selected
Top-K coordinates in chunks and reconstructs all parameters from streaming
statistics. The mathematical and falsifiability contract is in
`docs/ZI_SOHO_SPEC.md`.

`tools/zi_soho_pilot.py` is a dedicated train-only runner. It has no test
evaluation mode, optionally refuses to start while `test.pt` is visible, and
reads all method/search settings from
`configs/zi_soho_cifar100_train_only.json`. The fixed map is matched to the
locked FLY fidelity control (`H=10000`, degree `300`, coding level `0.3`, seed
`1993`). FLY keeps current-task GCV over exponents `[6,10)`; only ZI variance
shrinkage `kappa` is searched.

The runner caches sample-level sparse WTA codes on disk to avoid repeating the
fixed projection. That cache is explicitly labelled experiment
infrastructure, stored outside the output bundle and never serialized into a
learner checkpoint. Its expected CIFAR-100 size is roughly 0.9 GB.

## Gate and stopping rule

Selection uses a deterministic stratified 10% subset of training features.
The proposed hurdle model advances only if it is finite, beats raw Ridge by
0.20 percentage point, stays within 0.50 point of matched FLY, beats both
component scorers by 0.10 point, uses at most 15% of FLY persistent runtime
state, and leaves `test.pt` physically hidden.

Passing produces `REVIEW_FOR_HELDOUT_AUTHORIZATION`, not automatic permission
to evaluate test. Failure produces `STOP_TRAIN_ONLY_GATE_FAILED`. CIFAR-100 is
exploratory because its test split informed earlier work in this repository.

## Verification

Exact focused command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_zi_soho_math.py tests/test_zi_soho_learner.py tests/test_zi_soho_pilot.py
```

Result: `21 passed`, exit code `0`, pytest runtime `6.33s`. Warnings were 18
PyTorch JIT deprecations and one sparse-CSC beta warning. No warning was
suppressed and no test failed.

Exact full-suite command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `131 passed`, exit code `0`, pytest runtime `13.00s`. Warnings were 18
PyTorch JIT deprecations, one sparse-CSC beta warning and one sparse-invariant
warning. No warning was suppressed and no test failed.

The tests establish streaming/batch statistic equality, partition invariance,
direct scorer equivalence, chunk invariance, deterministic fixed projection,
checkpoint/logit round-trip, bounded state shapes, physical held-out locking,
candidate resumption, WTA cache identity, and exact equality of the ZI and FLY
projection/Top-K maps on synthetic data. They do not establish real-data
accuracy or validate the conditional-independence model.

## Colab handoff

Use `notebooks/zi_soho_cifar100_train_only_colab.ipynb`. Edit paths only in
Cell 2 and run all cells in order. Its progress is intentionally compact:

```text
WTA CACHE 12800/50000 (25.6%) elapsed=... eta=...
START 5/10 method=active_gaussian__variance_kappa-10p0
UPDATE method=active_gaussian task=4/10 stage_AA=... elapsed=...
DONE 5/10 method=active_gaussian ...
```

The notebook restores the frozen feature cache from Drive, extracts only if
missing, runs the focused tests, hides the held-out tensor, resumes completed
candidates from Drive, prints the locked gate, and downloads a train-only ZIP.
Do not restore/open `test.pt` or change the JSON config after viewing results.
