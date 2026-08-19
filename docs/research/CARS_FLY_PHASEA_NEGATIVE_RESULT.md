# CARS-FLY ImageNet-R Phase A result

Status: train-only feasibility study **FAIL**; held-out ImageNet-R test is not
authorized. This is a negative result, not a paper comparison.

The returned artifact was produced by clean commit
`f2617ec83a04c568e38adf289fa25ffa5e2794b1` with locked config
`configs/cars_fly_imagenetr_train_only.json` (SHA-256
`fd9b117751280aa3369f5db7408448cca4e9e90e82d41c4bfb94beb30e95508c`).
The ZIP SHA-256 is
`bed3ee45f549fc68b3781c28d6d88204569ddfb6ce30014f00853eaad31a0439`.
The run used Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8, seed 2025,
the verified frozen ViT checkpoint, and no held-out test features.

## Train-validation result

| Method | Validation AA (%) | Persistent state (bytes) |
|---|---:|---:|
| matched FLY | 77.9640 | 452,006,940 |
| raw Ridge | 76.0555 | 2,974,096 |
| Fisher residual | 73.5693 | 28,618,312 |
| full raw residual | 73.0004 | 40,933,528 |
| **CARS-FLY** | **72.3133** | **28,618,312** |
| fixed Schur residual | 72.2304 | 28,618,312 |
| compact anchor | 71.6326 | 27,649,368 |

CARS-FLY was `5.6507` percentage points below matched FLY, `3.7421` points
below raw Ridge, and `1.2559` points below the strongest low-rank control.
Its state was only `6.3314%` of matched FLY state. The numerical solve passed
with maximum relative residual `5.9184e-15`.

The selected rank schedule was
`[7, 15, 23, 31, 39, 46, 54, 61, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]`.
The energy threshold was reached only through task 8; the rank cap was active
from task 9 onward. Increasing or retuning that cap on the same validation
split would not be a clean confirmation and is not authorized by this phase.

## Interpretation

CARS-FLY demonstrates a real memory reduction but does not preserve the
predictive information of the full post-WTA FLY system. Its small advantage
over the fixed Schur control (`0.0830` point) is insufficient because it loses
to both matched FLY and raw Ridge. The adaptive-rank mechanism therefore does
not pass the accuracy gate on this protocol.

The useful design lesson is that compression before, or only through a
class-mean residual subspace after, the nonlinear Top-K representation drops
important covariance information. The next hypothesis must retain the full
FLY/WTA coordinates while compressing their second-order state, and it must
include a tail correction rather than treating discarded covariance as zero.

## Decision

- Stop CARS-FLY and do not evaluate the held-out ImageNet-R test.
- Preserve this artifact as negative evidence; do not overwrite it.
- Use ImageNet-R train validation only as a development protocol for the next
  hypothesis, since it has already informed method design.
- Require a new, separately locked dataset/protocol for any confirmatory claim.
