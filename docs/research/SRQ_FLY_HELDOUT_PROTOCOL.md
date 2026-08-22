# SRQ-FLY locked three-dataset held-out protocol

Status: **IMPLEMENTED AND AUTHORIZED BY PROJECT OWNER; REAL HELD-OUT RUN
PENDING.** Authorization permits exactly the immutable protocol below. It does
not permit changing a setting after observing test output.

## Question

Does groupwise-int8 square-root compression retain the predictor quality of
exact FLY-10,000 while using approximately the persistent tensor state of a
much narrower exact FLY model? The study reports all outcomes and has no
accuracy-based stopping or publication gate.

## Immutable identities

- manifest: `configs/srq_fly_three_dataset_heldout.json`;
- manifest SHA-256:
  `58036d4c282293eeca694d8e5f895bc260b4f8d6f13059c80cd60fac0aa2bd72`;
- held-out runner SHA-256:
  `4ef35968912f459579607c3eb4fbddd5d855a231fe6f34ef1ec9c1b2e9985d12`;
- held-out test extractor SHA-256:
  `09f97bfd9f96fc4f8dd93d00d92318fe307050db545debef02e3c54697c404c1`;
- notebook SHA-256:
  `78dc3a26189dd873a5ffa1b422871173a88a979899aa38d86b529b7ab76572f2`;
- frozen backbone:
  `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`;
- checkpoint SHA-256:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`;
- seeds and paired class/projection orders:
  `2025, 2026, 2027, 2028, 2029, 2030`.

The runner has no Ridge, rank, representation, seed, method-selection or
search command-line argument. Every such value comes from the hashed
manifest.

## Locked dataset settings

| Dataset | Tasks/classes | FLY/SRQ lambda | Raw lambda | Exact matched width |
|---|---:|---:|---:|---:|
| CIFAR-100 | 10/100 | `1e6` | `0.01` | 4,409 |
| CUB-200-2011 | 20/200 | `1e5` | `100` | 4,518 |
| ImageNet-R | 20/200 | `1e6` | `0.01` | 4,518 |

All methods use FLY width 10,000, synaptic degree 300 and coding level 0.3
except the explicitly state-matched exact-FLY control. SRQ uses block size 256,
group size 64, float32 streaming statistics and a float32 solver. Raw Ridge
uses float64 statistics and a linear solve.

The train-only evidence artifacts and hashes are embedded in the manifest.
The final notebook verifies their ZIP size, SHA-256, result status,
`uses_test_set=false` and selected lambdas before authorization.

Predeclared caveats are retained rather than repaired retrospectively:

- CIFAR selected `1e6` at the upper edge of its train-only grid. The project
  owner chose to freeze that value and skip a wider search; it may not be
  changed after test use.
- CUB D4 formally stopped because one seed missed the every-stage 98%
  prediction-agreement gate, although its accuracy/state hypotheses were
  supported. The stop remains visible in the manifest.
- all six seeds have been observed on training validation. Held-out examples
  were unseen, but these are algorithmic order/projection replications rather
  than fresh dataset samples.

## Single-use sequence

1. Clone a clean committed repository and verify manifest/runner hashes.
2. Download and hash the exact backbone checkpoint.
3. Audit CUB and ImageNet-R dataset identities without feature extraction.
4. Verify all completed train-only selection ZIPs.
5. Extract train features only; no `test.pt` may exist.
6. Run synthetic correctness/leakage tests.
7. Write `heldout_authorization.json` before held-out feature extraction.
8. Extract test features exactly once under that authorization.
9. Run all four methods for all six seeds on every dataset, without an
   accuracy-based early exit.
10. Aggregate every result and export a compact evidence ZIP. Feature and WTA
    caches are excluded.

Identical code/config may resume after infrastructure interruption. Numerical
or method failure is recorded as a result and does not authorize tuning.

## Endpoints

Per method, dataset and seed, the runner records:

- the triangular task-accuracy matrix;
- final accuracy, average incremental accuracy and forgetting;
- update and inference time;
- maximum solver relative residual;
- persistent tensor names, shapes, dtypes and bytes after every task;
- peak allocated GPU runtime memory;
- feature-cache and WTA-cache disk bytes.

The paired exact/SRQ unit also records prediction agreement and relative-logit
Frobenius error. The final report includes every seed, mean, sample standard
deviation and a two-sided 95% t interval. Runtime remains descriptive unless
the hardware/software environment is identical.

## State boundary

Persistent learner state may contain the sparse fixed projection, exact Gram
or compressed square-root factor, `Q`, counts, global classifier and bounded
metadata. It may not contain a historical image, feature, WTA code, label or
sample index. Per-sample feature/WTA files are explicitly reported as
experiment cache on disk and excluded from all exemplar-free checkpoints.

## Mandatory disclosure

The processed ImageNet-R artifact contains 19 byte-identical hashes across
train/test, including 18 under conflicting class directories. Its result must
always be labeled **legacy processed split**, not content-disjoint held-out
evaluation. CUB's audited split contains no cross-split duplicate content.
