# SRQ generalization M10: controlled GACL adapter

Status: PASS on the locked CIFAR-100 train-only controlled run. M10 does not
authorize test-feature extraction or a held-out test evaluation.

## Question

Can the square-root backend replace the inverse autocorrelation state used by
GACL when old and new classes coexist in the incoming stream?

The answer requires two distinct checks. First, the GACL inverse-RLS update
must be equivalent to a primal additive-Ridge system before quantization.
Second, P2B must retain useful validation accuracy and reduce total persistent
state when every method is updated at the same mini-batch frequency.

## Mathematical adapter

The GACL paper defines

\[
R_k=(X_{1:k}^{\top}X_{1:k}+\gamma I)^{-1}
\]

and updates it by Woodbury. Its classifier update is an inverse-RLS form that
handles both exposed and unexposed label columns. M10 instead maintains

\[
A_k=\gamma I+\sum_{j\le k}X_j^{\top}X_j,\qquad
B_k=\sum_{j\le k}X_j^{\top}Y_j,
\]

and solves `A_k W_k = B_k`. Expanding `B_k` when a new label first appears and
adding to an existing column when a class reappears is the primal counterpart
of GACL's exposed/unexposed decomposition. The local FP64 test checks the
inverse-RLS, primal, and joint solutions after every update.

SRQ is applied to the square-root factor of `A_k`; it is not applied directly
to GACL's inverse matrix `R_k`. Consequently, GACL's exact weight-invariance
property applies to the unquantized reference only. The P2B path is an
explicit approximation governed by the M8 perturbation analysis.

## Upstream lock and controlled scope

Primary upstream sources:

- repository: `https://github.com/CHEN-YIZHU/GACL`;
- commit: `fc21369792dd95a131692ae13190c763e0ed819c`;
- `methods/GACL.py` SHA-256:
  `9013fd5e2e59fc3bf37ea9c4b32a7367f05a0d5d83e642d9b0b6fb3ae4d62314`;
- `utils/online_sampler.py` SHA-256:
  `6a16c476bdda9d9dfc5bf006a6699725466a0ab729e8f4a46b5ecd1a0b03a7fb`;
- `models/pretrained.py` SHA-256:
  `a3ae8e342d7fe9f232461d2724d215077f7eb1802ca897c90b614168213a6ff2`;
- `scripts/gacl.sh` SHA-256:
  `a332c38045c384e2989c07883a2d63a80b1adb4956bbb2334f4c1e036410a21f`.

The official script uses a 5,000-dimensional bias-free random linear layer,
ReLU, `gamma=100`, five tasks, `N=50`, `M=10`, batch size 64, and
`online_iter=3`. M10 keeps width 5,000, ReLU, gamma, task count, N/M and batch
size, but locks `online_iter=1` so each selected sample has unit statistical
weight. Compression still occurs after every mini-batch.

This is not an end-to-end reproduction of the official benchmark. M10 uses
the existing frozen ViT-B/16 feature cache rather than the upstream
DeiT-small 384-dimensional checkpoint, does not reproduce image augmentation,
and selects 20 fit plus 5 validation samples per class from CIFAR training
data. The fixed-N/M branch of the upstream sampler is ported explicitly. These
scope differences must remain in every paper claim.

The current upstream `_Trainer` call also passes positional sampler arguments
in an order that makes the effective `varying_NM` behavior ambiguous relative
to the command-line flag. M10 therefore locks the fixed-N/M branch directly
rather than claiming byte-for-byte reproduction of that call site.

## Compared paths

All paths receive byte-identical expanded codes and identical mini-batches:

1. GACL inverse-RLS equation reference in FP32;
2. unquantized FP32 blocked-QR square-root;
3. FP16 square-root storage;
4. fixed P2B INT8/FP32 storage.

The inverse-RLS reference and the FP32 square-root are two numerical routes to
the same unquantized Ridge objective. P2B is compared to FP32 square-root for
accuracy retention and to it for total persistent-state reduction. Projection,
classifier, target statistic, count metadata, factor scales and all stored
tensors are included in total bytes. Feature caches are excluded and cannot be
deployed as learner state.

## Locked gates

- every selected fit sample appears in exactly one task;
- at least one class reappears across tasks;
- at least one mini-batch contains both previously exposed and newly exposed
  classes;
- all paths process the same number of mini-batch updates and rows;
- inverse-RLS versus FP32 square-root weight/logit errors remain below the
  committed tolerances and prediction agreement remains above its threshold;
- P2B reduces total persistent state by at least 70%;
- P2B validation-AIA loss relative to FP32 square-root is at most 0.5 point;
- the maximum analytic solver residual is at most `2e-5`.

Failure is a valid scientific result. No threshold, width, precision, gamma,
stream, or update frequency may be changed after observing M10 output.

## Run

Use `notebooks/srq_generalization_m10_gacl_colab.ipynb`. Run its cells from top
to bottom in a fresh T4 runtime. The notebook creates only `train.pt`, verifies
that `test.pt` is absent, runs local gates, executes the controlled stream, and
downloads:

`srq_generalization_m10_gacl_train_only.zip`

Return the ZIP whether the formal status is PASS or FAIL. Do not run a test
split in response to a failed development gate.

## Permitted conclusion

A PASS supports the statement that the backend is compatible with a
controlled GACL-style generalized stream at matched mini-batch frequency. It
does not establish reproduction of GACL's published accuracy, compatibility
with a changing backbone, or a universal plug-in claim. A later official
DeiT/checkpoint reproduction would still be required for the strongest GACL
claim.

## Recorded result

Artifact: `srq_generalization_m10_gacl_train_only.zip`; SHA-256
`8002ac80be9845186b8d9a753d0cf786f0e581b401c927dbf7a0dc95b713208c`.
The artifact records commit
`a2cdc3485bc7fb24fde90b5f7a47694473468258`, a clean checkout, no test-set
use, the locked upstream GACL identity, and
`PASS_M10_GACL_CONTROLLED_TRAIN_ONLY`. All ten declared gates pass.

| Path | Validation AIA | Final validation | Total state | Update time |
|---|---:|---:|---:|---:|
| GACL inverse-RLS FP32 | 84.8257 | 85.60 | 117,360,000 B | 0.779 s |
| FP32 square-root | 84.8257 | 85.60 | 119,360,400 B | 1.688 s |
| FP16 square-root | 84.8257 | 85.60 | 44,375,400 B | 2.646 s |
| P2B INT8/FP32 | 84.6869 | 85.40 | 32,658,996 B | 2.956 s |

The maximum inverse-RLS/FP32-square-root relative weight and logit errors are
`3.054e-5` and `1.002e-5`; minimum prediction agreement is `0.99999994`.
P2B reduces total state by 72.64% relative to FP32 square-root, with a 0.139
point validation-AIA loss. Its relative logit error rises from 0.0455 after
task 1 to 0.2075 after task 5, so the result supports compatibility but also
exposes cumulative approximation drift under mini-batch-frequency
compression.

The archived JSON has one reporting-only defect: `quadratic_persistent_bytes`
is zero for compressed backends because the collector recognized `factor_.*`
but not the actual `factor.*` tensor names. The authoritative total/backend
state fields and all formal gates are correct. Subtracting target statistic,
weights, and count metadata from the archived backend totals gives the true
factor payloads: 25,015,000 bytes for FP16 and 13,298,596 bytes for P2B. The
collector and regression test were corrected after artifact review.
