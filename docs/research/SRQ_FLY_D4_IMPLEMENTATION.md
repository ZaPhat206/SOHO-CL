# SRQ-FLY D4 implementation record

The real five-seed artifact has now been audited separately in
`docs/research/SRQ_FLY_D4_CUB_RESULTS.md`. Its formal status is
`STOP_SRQ_FLY_D4`; this implementation record does not relabel that outcome.

D4 adds an isolated runner/config/test/notebook layer and does not modify
existing FLY, SOHO, SRQ-FLY, feature-extractor, or dataset implementations.

The runner verifies the immutable D3 result and requires that its sole failed
gate is `numerical_stability`. It verifies five fresh nested splits, exact
projection prefixes, seed-specific sparse nonzero/state accounting, fixed D3
FLY lambda transfer, global train-only raw selection, resumable unit contexts,
and absence of `test.pt`. Projection state is computed from each seed's actual
stored sparse entries rather than incorrectly assuming every random projection
contains the same number of exact floating-point zeros.

Focused synthetic command (PowerShell, repository root):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_fly_d4_cub_multiseed.py
```

Result: `6 passed, 20 warnings in 23.88s`. The integration test covers five 20-task streams,
resume identity, nested train-only selection, D3 provenance, dynamic state
accounting, and visible-test refusal.

Related SRQ-FLY command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_fly_d4_cub_multiseed.py tests/test_srq_fly_d3_cub.py tests/test_srq_fly_learner.py tests/test_srq_fly_math.py
```

Result: `28 passed, 20 warnings in 27.42s`.

Full repository command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `253 passed, 20 warnings in 55.81s`. The warnings are the existing
PyTorch JIT deprecation and sparse CSC/invariant warnings; no test failed.

Notebook structural verification:

```powershell
python -m json.tool notebooks/srq_fly_cub_d4_multiseed_colab.ipynb
```

All eight notebook cells parse as valid JSON, and every code cell parses as
valid Python. No CUB held-out feature was created or opened by these local
tests.

The Colab notebook prints bounded raw-candidate START/DONE lines and live
CACHE/SEED/OUTER/TASK progress. It stores feature/WTA infrastructure outside
the evidence directory and safely resumes completed seed/method units.
