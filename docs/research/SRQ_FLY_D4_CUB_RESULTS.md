# SRQ-FLY D4 CUB multi-seed train-only result

Status: **formal `STOP_SRQ_FLY_D4`; primary accuracy/state hypotheses
supported**. D4 is train-only evidence. It does not authorize CUB held-out
evaluation and must not be relabeled as a pass.

## Immutable artifact

- artifact: `srq_fly_cub_d4_multiseed_train_only.zip`;
- artifact bytes: `102573`;
- artifact SHA-256:
  `6b65500e1d1dccb631f02d5c2016e55451f47762518a175dde7a40d6431c7fa1`;
- `d4_results.json` SHA-256:
  `713d7a31fa09a6f983ffd090bbb40c1be5c9e69a8585717cf3bc9a86451323ac`;
- locked config SHA-256:
  `a0742a545fa83f54b18bcf2372ea6d6e518d214f48f882f2b94696122c2fe8fd`;
- raw-lambda selection SHA-256:
  `100a83f2204c308bf304a7f6ebfc4e45e41c971f21dcfa07670bf20849c6f421`;
- CUB audit SHA-256:
  `a991b581ed8d78694f75381b9d862a242fcbfd6fe4ad366818161bf700d55125`;
- locked D3 result SHA-256:
  `f172d508c14fd95e7dcece5cd22c04a8e9f88c35c638fa140b476ebf6d4e6f4b`;
- clean runner commit:
  `2e4caa5b9895827178fd49948cefc7bbcf7f76a7`.

The environment was Python `3.12.13`, torch `2.11.0+cu128`, CUDA `12.8`,
and a Tesla T4. The ZIP contains 63 result/config/log entries and no feature or
WTA code cache. Those sample-level caches remained external experiment
infrastructure and are not learner state.

## Exact Colab command

The notebook invoked the following runner, with the displayed absolute paths
resolved from its locked cell 2 values:

```text
python -u tools/srq_fly_d4_cub_multiseed.py \
  --config configs/srq_fly_cub_d4_multiseed_train_only.json \
  --dataset-audit /content/cub_dataset_audit_d4.json \
  --feature-cache-dir /content/cub_train_feature_cache \
  --d3-result /content/locked_d3_results.json \
  --code-cache-root /content/drive/MyDrive/T-SOHO/srq_fly_cub_d4_wta_seed2026_2030 \
  --output-dir /content/drive/MyDrive/T-SOHO/srq_fly_cub_d4_multiseed_seed2026_2030 \
  --device cuda \
  --require-test-hidden
```

No CUB `test.pt` was present. Raw Ridge selected one global
`lambda=100` using inner training validation only; the selected value was not
at either end of the locked grid. FLY and SRQ used the transferred D3
`lambda=100000`, with no D4 retuning.

## Five-seed result

All values below are percentage points on the untouched outer train-validation
fold for each seed.

| Seed | Exact FLY-10000 AA | SRQ-FLY AA | FLY-4518 AA | Raw Ridge AA | SRQ minus FLY-4518 | Minimum agreement |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 90.1997 | 90.2416 | 90.1765 | 89.0928 | +0.0651 | 98.1396% |
| 2027 | 92.5860 | 92.5305 | 91.9258 | 90.4502 | +0.6046 | 97.8178% |
| 2028 | 90.6203 | 90.5909 | 89.3564 | 89.3842 | +1.2346 | 98.4165% |
| 2029 | 92.0328 | 91.9038 | 91.2196 | 89.6644 | +0.6843 | 98.4136% |
| 2030 | 92.5960 | 92.5828 | 92.5951 | 91.5276 | -0.0123 | 98.5261% |
| **Mean** | **91.6070** | **91.5699** | **91.0547** | **90.0239** | **+0.5153** | - |

Mean final accuracies were `87.2753`, `87.1086`, `86.5035`, and `83.9767`
for exact FLY-10000, SRQ-FLY, exact FLY-4518, and raw Ridge respectively.
SRQ's mean final gain over state-matched FLY was `+0.6051` point.

SRQ used approximately `105.167 MB` of persistent tensor state versus
`452.007 MB` for exact FLY-10000 and `105.150 MB` for exact FLY-4518. Its
state fraction of exact FLY-10000 was `23.27%`; its mismatch from the
state-matched control was below `0.1%` for every seed. Raw Ridge used about
`7.178 MB` and was lower accuracy, so it did not Pareto-dominate SRQ.

The mean SRQ gain over FLY-4518 was `+0.5153` point with sample standard
deviation `0.5086` and a two-sided 95% t interval of
`[-0.1162, +1.1467]`. SRQ won four of five seeds. The interval includes zero,
so D4 alone is not a significance claim.

## Gate interpretation

Eighteen of nineteen locked gates passed. Numerical stability passed:

- maximum raw-search residual: `1.7097e-14`, below `2e-5`;
- maximum reported outer residual: `7.7899e-6`, below `1e-5`.

The sole failed gate was `prediction_agreement_every_seed`. D4 required at
least 98% agreement with exact FLY-10000 at every stage of every seed. Seed
2027 reached `97.8178%` at task 20, missing the threshold by `0.1822`
percentage point. At that stage exact and SRQ accuracies were `86.7556%` and
`86.5904%`, a `0.1653`-point gap; the solve remained numerically valid.

The gate is not removed or weakened after observing this result. The failure
does not erase the observed accuracy/state ordering, but it prevents a claim
that SRQ reproduced the exact predictor at the preregistered fidelity level on
every seed.

## Limits

- D4 used training-derived outer validation, not CUB held-out test.
- Five seeds do not establish a nonzero population gain at 95% confidence.
- The frozen ViT cache and WTA caches contain sample-level features/codes and
  are experiment infrastructure, not deployable exemplar-free state.
- Runtime numbers from Colab are not paper-hardware comparisons.
- D4 does not test error feedback, lower-bit factors, a quantized backbone, or
  train-from-scratch continual learning.
