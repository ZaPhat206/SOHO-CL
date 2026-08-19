# TAIL-FLY ImageNet-R Phase A result

Status: train-only development gate **FAIL**. Held-out ImageNet-R test is not
authorized. This artifact is immutable negative evidence, not a paper result.

## Artifact identity

- ZIP: `tail_fly_imagenetr_phasea_train_only.zip`
- ZIP SHA-256:
  `c24e77521d7035eb8cb9bf13954d8369016ab6e1f445f8d94159e61a2c9d6987`
- clean Git commit: `e0571ce69ec52f92e2ab96adf536f8da07bbaa55`
- locked config SHA-256:
  `c49b6a7e813c94d40413dd2d4f8e5e7889fff9c5b1aea0a5e7af046c0913bc04`
- shared unit context SHA-256:
  `aba7fab50679a6dce7f03dec362469881ad370d86b352e736833e4ecbb29e9fd`
- seed: `2025`
- environment: Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8
- `uses_test_set=false`, `held_out_test_authorized=false`, and `test.pt` was
  not visible.

## Best independently selected train-validation configurations

| Method | Rank | Ridge | Validation AA (%) | Final-stage validation AA (%) | Resident state |
|---|---:|---:|---:|---:|---:|
| matched exact FLY | 10000 | task-GCV, final `1e6` | 77.9640 | 71.1445 | 452,006,940 B |
| raw Ridge | 0 | `0.1` | 76.0555 | 69.3215 | 3,588,896 B |
| plain TSVD-FLY | 256 | `1e4` | 73.8327 | 62.9919 | 62,247,964 B |
| **TAIL-FLY** | **64** | **`1e8`** | **71.4428** | **65.3538** | **54,607,196 B** |
| diagonal-only FLY | 0 | `1e4` | 10.1912 | 5.0679 | 52,046,940 B |

TAIL-FLY was `6.5213` percentage points below matched exact FLY, `4.6127`
points below raw Ridge, and `2.3900` points below independently selected plain
TSVD-FLY. Its resident state was `12.0811%` of exact FLY state, an approximately
87.9% reduction.

At TAIL-FLY's selected rank and Ridge value, it exceeded the same-configuration
plain TSVD control by `18.9401` points. This validates that the diagonal tail is
not a no-op under severe truncation, but it does **not** establish superiority
to independently selected plain TSVD. The original gate exposed only the former
comparison and must not be presented as overall control superiority.

## Gate outcome

| Gate | Observed | Decision |
|---|---:|---|
| exact test remained hidden | true | PASS |
| resident state / exact FLY | `0.12081` | PASS (`<=0.25`) |
| same-configuration tail gain | `+18.9401 pp` | PASS (`>=0.20`) |
| gap to matched exact FLY | `6.5213 pp` | **FAIL** (`>0.50`) |
| gain over raw Ridge | `-4.6127 pp` | **FAIL** (`<0`) |
| maximum reported solver residual | `3.1599e-3` | **FAIL** (`>1e-5`) |

The residual aggregate did not identify which method/rank/Ridge produced the
maximum. Float32 QR/SVD was intentionally used for T4 feasibility, but the
Woodbury solve and residual check also remained float32. Phase A3 may correct
only this numerical/instrumentation defect. It must not change the data split,
seed, rank grid, Ridge grid, representation, or gates after observing these
results.

## Interpretation and stopping boundary

The exact diagonal tail recovers coordinate-wise energy but not discarded
off-diagonal covariance. Plain TSVD improved monotonically with rank, whereas
TAIL-FLY did not improve beyond rank 64. This supports the interpretation that
the missing off-diagonal tail contains predictive information that a diagonal
correction cannot reconstruct.

Phase A3 is a correction run on the same development split:

1. keep streamed QR/SVD/statistics in float32;
2. perform only the reduced analytic solve and residual verification in
   float64;
3. report residuals separately for every method, rank, Ridge, and task;
4. bind resumable artifacts to Git/code identity;
5. compare independently selected TAIL-FLY and plain TSVD in addition to the
   matched-configuration ablation.

If numerical correctness passes but TAIL-FLY remains below raw Ridge or the
independently selected plain TSVD control, close the direction. Do not switch
seed or dataset to conceal the negative result and do not evaluate held-out
test.
