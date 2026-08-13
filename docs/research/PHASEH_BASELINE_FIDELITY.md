# Phase H-A — matched baseline fidelity gate

Status: implementation and synthetic fidelity gate **PASS**. No CIFAR-100
experiment was run in this phase.

## Purpose

Phase G supports Schur Residual SOHO on one locked CIFAR-100 class order, but
the earlier cache adapters were not faithful reference reproductions. They
used a fixed Ridge coefficient and a compact seen-class target, whereas the
current source implementations use a fixed global class output and internal
GCV policies. Phase H-A fixes only the cache adapters. Original
`methods/flycl.py`, `methods/sohocl.py`, `models/flyhash.py`, and
`models/soho.py` remain unchanged.

## Explicit methods

`cached_flycl_fidelity` matches current FLY source semantics:

- the fixed sparse Gaussian FlyHash projection and positive WTA;
- a fixed `num_classes`-wide target and classifier;
- cumulative projected-feature `G,Q`;
- current-task GCV over the declared exponent range;
- no historical feature/sample state.

`cached_soho_replay_fidelity` matches current SOHO source semantics:

- dynamic spherical OLDA, ETF/Procrustes, sparse Rademacher expansion and WTA;
- exact replay/reprojection of all historical backbone features;
- a fixed `num_classes`-wide target and classifier;
- current SOHO replay-wide random GCV sample policy;
- serialized replay features, replay labels, projection RNG continuation and
  OLDA state.

SOHO fidelity is explicitly **not exemplar-free**. Its sample-level replay is
included in persistent-state bytes. The feature cache itself remains shared
experiment infrastructure and is reported separately.

Legacy `cached_flycl` and `cached_soho_replay` names are preserved so prior
outputs remain reproducible. The fidelity methods use separate config names
and are rejected from the runner's external train-validation search grid;
their Ridge choices come only from the locked internal GCV policy.

## Locked CIFAR-100 configuration

- FLY: `expand_dim=10000`, synaptic degree `300`, coding level `0.3`, GCV
  exponents `[6,10)`.
- SOHO: `expand_dim=10000`, density `0.1`, OLDA dimension `768`, ETF on,
  coding level `0.25`, GCV exponents `[-2,10)`, replay chunk `2000`, GCV sample
  cap `3000`.
- Shared frozen ViT checkpoint SHA-256:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`.

These values come from the repository's current baseline scripts/profiles;
they are not selected from Phase G held-out accuracy.

`configs/phaseh_cifar100_multiseed.json` preregisters seeds
`1993,2025,3407,4421,5501`, the frozen Schur parameters and paired reporting.
It is a manifest only: this commit does not implement or execute the
multi-seed study.

## Fidelity invariants

Synthetic tests instantiate the original and cache-native learners from the
same RNG state and fixed feature batches. They require equality of projection,
selected Ridge per task, accumulated `G,Q`, final classifier, logits and
predictions within the stated float32 tolerances. Resume tests additionally
require continued updates to remain deterministic. Runner tests verify
explicit dispatch, exemplar disclosure and rejection of external search.

The synthetic comparison validates adapter equivalence for the exercised
stream and source revision. It does not reproduce the published FLY accuracy,
estimate multi-seed variance, validate runtime parity, or make SOHO
exemplar-free.

## Exact commands and results

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/test_cached_replay_baselines.py tests/test_experiment_runner.py tests/test_schur_locked_eval.py -q
```

Result: `21 passed`, exit code `0`, pytest runtime `6.68s`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `73 passed`, exit code `0`, pytest runtime `8.12s`.

Warnings were 18 PyTorch JIT deprecations, one sparse-CSC beta warning and one
sparse-invariant warning. No warning was suppressed and no test failed.

## Next gate

Only after focused and full tests pass may a separate Phase H-B runner and
Colab notebook execute the preregistered five-seed study. This original
stopping proposal was superseded before Phase H execution by the transparent
amendment in `PHASEH_MULTISEED_RUNBOOK.md`: the `93.89` paper value remains a
reported reproduction diagnostic but does not stop the matched internal
comparison.
