# SRQ-FLY D3: locked CUB train-only replication protocol

Status: preregistered implementation protocol. No CUB held-out evaluation is
authorized by this document.

## Question and interpretation

D2.1 established one positive ImageNet-R train-validation result: an int8
square-root FLY system with width 10,000 used approximately the same persistent
state as exact FLY width 4,518, while improving average incremental validation
accuracy by 0.9201 percentage point. D3 asks whether that **state-efficiency
ordering** transfers to fine-grained CUB-200. It is not a claim that SRQ-FLY is
universally better than FLY.

Four methods are mandatory:

1. exact FLY-10,000, the uncompressed accuracy ceiling;
2. SRQ-FLY-10,000 with groupwise-int8 storage of the square-root factor;
3. exact FLY-4,518, the closest exact-FLY state budget that does not exceed
   SRQ-FLY-10,000;
4. streaming raw-feature Ridge, an explicit low-state sanity baseline.

All use the same frozen ViT features, class order, task split, seed, and
evaluation indices. FLY-4,518 uses the first 4,518 rows of the same seeded
projection as FLY-10,000. Its Top-K code is recomputed at width 4,518; the
protocol does **not** claim that dynamic Top-K can be linearly transported.

## Locked data and split

- dataset: processed `CUB-200-2011` from Kaggle
  `zaphat206/cub-200-2011`;
- raw-data identity SHA-256:
  `e374af9b576cb6b3503198ef3ea30fd0aa9d2e18c230ff8064e21d4f644af2ca`;
- train/test counts: 5,994/5,794; 200 classes; no content duplicates across
  the two directory splits;
- frozen backbone: `vit_base_patch16_224` with checkpoint
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`;
- preprocessing: resize 256, center crop 224, normalize by mean/std 0.5;
- seed: `2025`; 20 tasks of 10 classes; no Task-ID at inference;
- outer validation: deterministic stratified 20% of CUB train;
- inner validation: deterministic stratified 20% of each outer-training
  partition.

The inner split alone chooses hyperparameters. Outer validation is evaluated
once after selection. `test.pt` must not exist, and the runner has no test
evaluation mode. Dataset audit may hash the held-out files to verify identity;
no held-out image is passed through the backbone and no held-out label or
feature enters model selection.

## Locked selection

The FLY Ridge grid is
`1e4, 1e5, 1e6, 1e7, 1e8, 1e9`. Exact FLY-10,000 selects one value on the
inner split; SRQ-FLY-10,000 receives that same value unchanged so its comparison
to the exact system isolates storage approximation. Exact FLY-4,518 selects its
own value on the same inner split because it is the independently optimized
state-matched competitor. Raw Ridge uses float64 sufficient statistics for the
known CUB Cholesky-conditioning issue and selects from
`0.001, 0.01, 0.1, 1, 10`. Ties choose the smaller lambda.

The locked machine-readable configuration is
`configs/srq_fly_cub_d3_train_only.json`, SHA-256
`26926c625aa6dbebf7271a4767d73a56e9bfe3bf0e139edca012977789dec772`.

## Persistent state and gates

Persistent learner state contains only the fixed sparse projection, streaming
cross-statistics/counts, classifier, and either the exact Gram or compressed
square-root factor. It contains no images, per-sample frozen features, WTA
codes, or historical labels. Feature/WTA caches are experiment infrastructure
on disk and are excluded from evidence checkpoints.

At 200 seen classes, locked runtime tensor accounting is:

- exact FLY-10,000: 452,006,940 bytes;
- SRQ-FLY-10,000 int8: 105,166,628 bytes;
- exact FLY-4,518: 105,149,848 bytes.

The 12-byte difference from nominal width-10,000 formulas is preregistered: the
seeded sparse projection has 2,999,999 stored nonzeros rather than 3,000,000.

D3 passes for review only when all integrity/numerical gates pass, SRQ stays
within 0.50 point of exact FLY-10,000 in outer average and final accuracy,
paired prediction agreement is at least 98%, SRQ state is at most 25% of exact
FLY-10,000, state mismatch versus FLY-4,518 is at most 0.1%, and SRQ improves
over independently tuned FLY-4,518 by at least 0.10 point in average accuracy
and 0.00 point in final accuracy. The report also fails its research gate if
raw Ridge Pareto-dominates SRQ in outer average accuracy and persistent bytes.

A pass authorizes review only. A failure is a valid negative replication and
must not be repaired by changing seed/grid after viewing outer validation.
