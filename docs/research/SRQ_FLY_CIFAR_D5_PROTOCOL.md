# SRQ-FLY D5: CIFAR-100 train-only selection protocol

Status: **real train-only execution completed with `PASS_REVIEW_CIFAR_D5`;
the separately locked three-dataset protocol now governs held-out use.**

## Purpose

SRQ-FLY already has train-only Ridge choices for CUB and ImageNet-R. D5 fills
the missing CIFAR-100 choice without opening `test.pt`. The resulting artifact
is a prerequisite for the later three-dataset held-out protocol.

The locked comparison contains exact FLY-10,000, SRQ-FLY-10,000, exact
state-matched FLY-4,409 and float64 raw Ridge. For 100 classes, `m=4,409` is
the closest exact-FLY dimension whose nominal persistent tensor state does not
exceed SRQ-FLY's `97,166,240` bytes. `m=4,410` exceeds that budget.

## Selection and controls

- seed and class order: `2025`;
- ten class-incremental tasks over 100 classes;
- outer validation fraction: `0.20`;
- inner validation fraction: `0.20` of outer-fit;
- FLY/SRQ lambda grid: `100, 1e3, 1e4, 1e5, 1e6`;
- tie-break: maximum inner average incremental accuracy, then smaller lambda;
- a candidate whose unchanged float32 system fails the fixed-Ridge Cholesky
  check is recorded as `solver_failed` and is ineligible for selection; the
  runner must not add jitter, change its lambda, or silence any other runtime
  error;
- selected lambda is shared by exact FLY-10,000, SRQ-FLY-10,000 and
  FLY-4,409;
- raw Ridge uses the existing locked CIFAR value `0.01` from the Phase H
  protocol and is not retuned here.

The outer validation partition is used only once after selection. The test
cache must be absent, and the result always records
`held_out_test_authorized=false`.

The failure-handling rule above was added after the first real D5 invocation
encountered a Cholesky exception at `lambda=100`. That invocation exposed no
candidate accuracy and never reached outer validation. The locked grid was
not changed; the failed value remains in the evidence artifact.

## Locked identities

- config: `configs/srq_fly_cifar100_d5_train_only.json`;
- config SHA-256:
  `9af36b234d980962e6834a9ccd9f5204c9c7f660e44a5faaf9c6332e3151ea81`;
- backbone checkpoint SHA-256:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`;
- feature dimension: `768`;
- SRQ storage: block size `256`, group size `64`, float32 statistics and
  solver.

## Real command

The Colab notebook executes:

```bash
python -u tools/srq_fly_cifar_selection.py \
  --config configs/srq_fly_cifar100_d5_train_only.json \
  --feature-cache-dir /content/srq_cifar_d5_train_cache \
  --large-code-cache-dir <DRIVE>/srq_cifar_d5_wta_m10000_seed2025 \
  --matched-code-cache-dir <DRIVE>/srq_cifar_d5_wta_m4409_seed2025 \
  --output-dir <DRIVE>/srq_cifar_d5_outputs \
  --device cuda
```

The local cache must contain only `metadata.json` and `train.pt`. WTA caches
are sample-level experiment infrastructure and are not learner state.

## Interpretation gate

A `PASS_REVIEW_CIFAR_D5` means selection, numerical, paired-fidelity and state
accounting checks passed. It does not mean SRQ beats FLY and does not authorize
test evaluation by itself. The reviewed ZIP SHA-256 is
`fca77cd948bcb1ad59ea52efd71d79fad42d8de763559177e15aea206a8d2ca9`;
it selected `lambda=1e6`. A stop result is returned unchanged for review;
hyperparameters must not be edited after seeing outer-validation metrics.
