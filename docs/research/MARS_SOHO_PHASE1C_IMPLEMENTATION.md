# MARS-SOHO Phase 1C implementation

Phase 1C is isolated in `methods/mars_soho/tangent.py` and
`tools/mars_soho_phase1c.py`. Existing SOHO, FLY, SRQ and completed MARS modes
are unchanged.

The tangent module implements spherical log/exp maps, deterministic
class-specific randomized covariance sketches, exact diagonal residuals,
antithetic reconstruction and optional resultant-length calibration. The
runner refuses visible test caches, selects rank only on nested inner
validation, evaluates four locked controls on outer train-validation, audits
aggregate state and resumes units by source/context hash.

The implementation deliberately stops before integrating the sketch into the
continual learner. This prevents a long continual experiment until direct
hard-WTA `G,Q` fidelity has passed.

## Local verification

Environment: Windows, Python 3.13, CPU PyTorch. No dataset, checkpoint or
held-out test split was opened by these commands.

```text
python -m pytest -q tests/test_mars_soho_math.py tests/test_mars_soho_learner.py tests/test_mars_soho_phase1.py tests/test_mars_soho_phase1b.py tests/test_mars_soho_tangent.py tests/test_mars_soho_phase1c.py
```

Result after the final oracle-state assertion was added: `32 passed in 8.63s`.

```text
python -m pytest -q
```

Final result: `340 passed, 20 warnings in 62.53s`. The warnings are
pre-existing PyTorch JIT deprecations and sparse CSC/invariant notices; no
Phase-1C test failed.

The Colab notebook embeds these immutable identities:

```text
config SHA-256: 9ca4940b50d7ebc7560ce65e0ddcf00e8ead47adb5e0d626fbd1fac0e7461da0
runner SHA-256: ce4f22a9bbd5075484303879a6bffeff64eea5b3750ea1772b20e72ace517548
```
