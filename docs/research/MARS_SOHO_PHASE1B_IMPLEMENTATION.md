# MARS-SOHO Phase 1B implementation

## Scope

Phase 1B changes only the allocation signal in the isolated MARS-SOHO research
namespace. Existing SOHO, FLY and SRQ implementations are unchanged. The
binary support-aware Phase-1 modes remain available so the completed artifact
retains an explicit semantic reference.

New implementation pieces:

- `topk_support_turnover`: exact realized Top-K replacement fraction;
- `wta_statistic_variance`: matrix-free relative variance estimator for
  `(zz^T,z)`;
- disjoint deterministic pilot streams;
- turnover/statistic-variance and shuffled learner modes;
- `tools/mars_soho_phase1b.py`: fresh, fail-closed train-only runner;
- `configs/mars_soho_phase1b_train_only.json`: immutable settings and gates;
- `tests/test_mars_soho_phase1b.py`: lock, test-hiding and resume checks.

The runner performs no search. It uses Phase-1 inner-selected values and fresh
train-only validation seeds, runs six paired controls, records per-stage risk
spread/allocation diagnostics, and writes `phase1b_results.json` plus resumable
per-method units.

## Interpretation

The primary scientific comparison is statistic-variance allocation versus
uniform and shuffled-statistic-variance controls. Turnover methods diagnose
whether continuous support movement is informative, but cannot be promoted to
the primary method after results are observed.

Persistent learner state remains counts, sums, squared sums, pooled scatter,
the current SOHO map, analytic Gram/cross and classifier. Pilot and pseudo
samples are temporary update-time tensors and never enter the checkpoint.

## Verification

Focused mathematical, learner, historical Phase-1 and Phase-1B runner tests:

```text
python -m pytest -q tests/test_mars_soho_math.py tests/test_mars_soho_learner.py tests/test_mars_soho_phase1.py tests/test_mars_soho_phase1b.py
```

Result: `27 passed in 7.99s`.

Full repository regression:

```text
python -m pytest -q
```

Result: `335 passed, 20 warnings in 62.86s`. The warnings are existing
TorchScript deprecations and PyTorch sparse CSC/invariant warnings; none
originates from the MARS-SOHO namespace.

Direct entrypoint:

```text
python tools/mars_soho_phase1b.py --help
```

Result: exit code `0`.
