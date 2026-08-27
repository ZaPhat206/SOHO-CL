# MARS-SOHO Phase 1 CIFAR-100 result

## Decision

Phase 1 is a valid train-only negative result. The implementation and
exemplar-free state audits passed, but the scientific gate failed. Do not open
the CIFAR-100 test cache and do not add SRQ to this Phase-1 method.

Artifact: `mars_soho_phase1_cifar100_train_only.zip`

```text
SHA-256: d1241647c32617d3858cc1964b3b967285a9e8ec0a0320ed4d5dd31a36af72f0
git commit: b8956eafc0b31ed780a080cc7fcf85ea56c5c9da
Python: 3.13.15
torch: 2.11.0+cu128
GPU: Tesla T4
uses_test_set: false
```

The locked train-only selection chose Ridge `lambda=10`, covariance rank `64`,
shrinkage `0.1`, and `64` pseudo-directions per old class. Results below are
mean +/- sample standard deviation over three class-order/projection
replicates.

| Method | AIA (%) | Final (%) | Forgetting (%) | Persistent state |
|---|---:|---:|---:|---:|
| exact replay oracle | 90.925 +/- 0.430 | 85.620 +/- 0.182 | 5.593 | 136,408,464 B |
| shared Gaussian | 89.330 +/- 0.447 | 83.760 +/- 0.089 | 8.667 | 13,208,464 B |
| heterogeneous spherical | 89.315 +/- 0.427 | 83.700 +/- 0.113 | 8.726 | 13,208,464 B |
| support aware | 89.315 +/- 0.427 | 83.700 +/- 0.113 | 8.726 | 13,208,464 B |
| shuffled support | 89.315 +/- 0.427 | 83.700 +/- 0.113 | 8.726 | 13,208,464 B |

The exemplar-free state is 90.32% smaller than the matched oracle state in
this feasibility configuration, and contains zero historical feature/label
rows. This is persistent learner state, not peak runtime memory.

## Gates

| Gate | Threshold | Observed | Decision |
|---|---:|---:|---|
| support gap to oracle | at most 0.50 pp | 1.610 pp | fail |
| support gain over shared Gaussian | at least 0.20 pp | -0.014 pp | fail |
| support gain over shuffled support | at least 0.10 pp | 0.000 pp | fail |
| test cache hidden | required | true | pass |

Across all 1,350 recorded old-class/task/replicate diagnostics, certificate
failure risk was exactly `1.0`; all allocations were exactly `64`. Therefore
support-aware, shuffled-support and heterogeneous-uniform replay were the same
estimator in this run. The support theorem remains correct, but its sufficient
condition is too conservative to rank replay effort under the observed SOHO
map changes.

Phase 1 does not prove moment replay is impossible. It falsifies the specific
binary-certificate allocation mechanism at width 1,000 on the consumed
CIFAR-100 train-only validation protocol.
