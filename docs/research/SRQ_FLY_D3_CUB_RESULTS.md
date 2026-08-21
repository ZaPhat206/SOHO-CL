# SRQ-FLY D3 CUB train-only result

Status: **formal `STOP_SRQ_FLY_D3`; substantive state/accuracy hypotheses
supported**. This is not a held-out result and does not authorize CUB test
evaluation.

## Immutable artifact

- ZIP: `srq_fly_cub_d3_train_only.zip`;
- ZIP SHA-256:
  `4d2104c80e3f5fa125839f7723ac86126fbb0395c53b7307a9de1a349b8f380a`;
- result SHA-256:
  `f172d508c14fd95e7dcece5cd22c04a8e9f88c35c638fa140b476ebf6d4e6f4b`;
- locked config SHA-256:
  `26926c625aa6dbebf7271a4767d73a56e9bfe3bf0e139edca012977789dec772`;
- runner commit: `4cb65d756febd3e90382c47b236a7031462ede4a` (clean);
- train feature SHA-256:
  `b58dd541f35450d926e1c82826d6491665ca55920a0d5627e2153b65d2015688`;
- environment: Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8, Tesla T4;
- seed 2025; 20 tasks; nested train-only selection; `test.pt` absent.

The evidence ZIP contains JSON/config/log files only. Frozen per-sample train
features and WTA codes remained external experiment caches and are not learner
state or evidence-checkpoint contents.

## Locked outer-validation results

| Method | Ridge | Average incremental accuracy | Final accuracy | Persistent state |
|---|---:|---:|---:|---:|
| SRQ-FLY-10000 int8 | 1e5 | **91.7564** | **87.5282** | 105,166,628 B |
| Exact FLY-10000 | 1e5 | 91.6761 | 87.1102 | 452,006,940 B |
| Exact FLY-4518 | 1e5 | 91.5147 | 86.9393 | 105,149,848 B |
| Raw Ridge float64 | 10 | 89.6107 | 84.8446 | 7,177,792 B |

SRQ-FLY improved over the state-matched exact FLY-4518 by 0.2417 percentage
point in average incremental accuracy and 0.5890 point in final accuracy. It
used 23.2666% of exact FLY-10000 state and differed from FLY-4518 state by only
0.01596%. Minimum paired agreement with exact FLY-10000 was 98.2303%. On this
single seed, SRQ also exceeded exact FLY-10000 by 0.0803 average and 0.4181
final point; this must not be generalized as a quantization benefit.

## Why the formal status is STOP

Twelve of thirteen gates passed. The only failed gate aggregated numerical
residuals over **all** inner candidates. The rejected `lambda=1e4` candidates
had maximum relative residuals `1.14014e-5` at width 10,000 and `1.04537e-5`
at width 4,518, slightly above the preregistered `1e-5` limit. The selected
inner candidates and every outer method were below `1e-5`; there was no solver
failure, NaN, Inf, traceback, state mismatch, or test leakage.

The gate is not changed retrospectively. D3 remains a formal STOP with a
positive scientific signal. A prospective phase may define separate tolerances
for discarded search candidates and reported outer models, but it cannot
rename D3 as PASS.

## Limitations carried forward

- one CUB seed and train-validation only;
- raw Ridge selected the upper grid boundary `lambda=10`, so its optimum was
  not bracketed;
- the positive difference from exact FLY-10000 may be seed-specific implicit
  regularization;
- no claim about held-out accuracy, multi-seed significance, INT8 backbone
  speed, or universal superiority is supported.
