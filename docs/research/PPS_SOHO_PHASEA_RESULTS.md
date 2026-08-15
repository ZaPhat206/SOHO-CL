# PPS-SOHO Phase A train-only results

Status: numerical gate passed; overall Phase A gate failed. CIFAR-100 held-out
test evaluation is not authorized.

## Evidence

- Artifact: `pps_soho_phasea_train_only.zip`
- Artifact size: 1,561 bytes
- Artifact SHA-256:
  `0718ffc0f2434e5c140cdd121570f5ebb427eea3fdcbc071a623cad33a73d1a8`
- Contents: `selection.json` only (10,484 bytes)
- Protocol recorded by runner: stratified held-out subset of cached training
  features only
- Candidate count: 16; every candidate records `uses_test_set=false`
- Seed: 1993
- Statistics dtype: float32 storage with promoted compact solve/classifier
- Solver tolerance: maximum relative residual at most `1e-4`

Exact runner command from the notebook:

```text
python -u tools/experiment_runner.py --select-config --config configs/pps_soho_cifar100_pilot.json --feature-cache-dir /content/tsoho_cifar100_cache --output-dir /content/pps_soho_phasea_outputs/selection --selection-output /content/pps_soho_phasea_outputs/selection.json --device cuda
```

The cache records the frozen ViT checkpoint SHA-256
`32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`,
768-dimensional finite float32 features, 50,000 training samples, and 10,000
test samples. Its embedded Git commit identifies cache extraction provenance,
not the selection runner.

## Best train-validation results

| Method | Rank | Lambda | Gamma | Validation AA (%) | State bytes |
|---|---:|---:|---:|---:|---:|
| Sufficient-statistic raw Ridge | 0 | 0.1 | - | 89.3964 | 5,948,192 |
| Matched cached FLY | 0 | 0.1 | - | 88.6073 | 8,706,456 |
| PPS class-protected | 32 | 1.0 | 1.0 | 83.2073 | 5,052,824 |
| PPS standard FD | 16 | 1.0 | - | 75.6836 | 4,987,288 |

All selected PPS residuals were below `1e-4`; the class-protected winner's
relative residual was approximately `6.81e-15`.

## Interpretation

- Class protection improves over standard FD by 7.5237 percentage points at
  nearly equal memory. This supports the narrow claim that class-aware moment
  preservation is useful under aggressive sketching.
- PPS remains 5.4000 points below matched FLY, so it fails the primary accuracy
  gate.
- Raw Ridge exceeds PPS by 6.1891 points and FLY by 0.7891 points. It remains
  the mandatory strong baseline.
- PPS uses about 42% less persistent state than matched FLY, but Phase A does
  not establish a competitive accuracy-memory frontier.
- Increasing rank from 16 to 32 improves class-protected PPS by only 0.08
  points, despite a substantially smaller covariance error bound. A covariance
  approximation guarantee is not a classifier-accuracy guarantee.

The artifact lacks an exact selection-runner commit, dirty-worktree status,
environment identity, class-order hash, and validation-index hash. Therefore
the values are sufficient for the Phase A decision but not a final paper
artifact. Phase A2 records those fields explicitly.
