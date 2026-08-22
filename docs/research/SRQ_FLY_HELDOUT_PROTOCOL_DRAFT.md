# SRQ-FLY held-out protocol draft

> **Superseded historical draft.** The active project-owner-authorized
> protocol, immutable identities and executable notebook are recorded in
> `SRQ_FLY_HELDOUT_PROTOCOL.md`. Pending values and blockers below describe
> the earlier review state and are retained only as an audit trail.

Status: **DRAFT FOR INDEPENDENT REVIEW — HELD-OUT EVALUATION IS NOT
AUTHORIZED.** This file creates no experiment permission and no executable
test runner. CUB and ImageNet-R test features must remain unopened.

## Purpose

The proposed single-use three-dataset study tests whether the train-validation
accuracy/state signal survives on held-out examples after all method choices
are frozen. It does not redefine the formal D3 or D4 stops, and it has no
accuracy-based publication gate: a negative result must still be reported.

## Frozen method identity

The method is the existing groupwise-int8 square-root learner with no error
feedback and no lower-bit variant:

- base implementation commit: `bde8caddabb74ac43db3ff759933bc976ec563cb`;
- `methods/srq_fly/learner.py` SHA-256:
  `5a1ee62022b98f6334c0c371087ffeb6512699f99c36fa4057db40c6219ac206`;
- `methods/srq_fly/storage.py` SHA-256:
  `c095ecf98bdd991950998e41018c2b2a0ce13b7d16586cdac14143cea7c16c25`;
- `models/flyhash.py` SHA-256:
  `24ba321a71f735031b0da430ab4d3519e54e6c3149fc6c913c63b4172f6712cb`.

Changing these files creates a new method and requires a new train-only study.
Documentation-only paper commits do not change method identity.

## Shared representation contract

- backbone: `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`;
- checkpoint SHA-256:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`;
- frozen pooled feature dimension: `768`;
- preprocessing: resize 256, center crop 224, tensor conversion, normalization
  with mean `(0.5, 0.5, 0.5)` and std `(0.5, 0.5, 0.5)`;
- 20 class-incremental tasks over 200 global classes;
- no task ID at inference;
- FLY expansion: `m=10000`, synaptic degree `300`, coding level `0.3`;
- state-matched FLY expansion: `m=4518`, otherwise identical;
- for each seed, FLY-4518 uses exactly the first 4,518 rows of the paired
  FLY-10000 projection, verified before evaluation;
- SRQ block size `256`, group size `64`, float32 statistics and solver;
- seeds/class-order/projection seeds: `2025, 2026, 2027, 2028, 2029, 2030`.

These seeds have been observed on train-validation and must not be called
fresh independent replications. Their held-out examples remain unseen; the
study estimates sensitivity to fixed class-order/projection choices, not
independent dataset sampling.

## Dataset identity preconditions

CIFAR-100 requires the separately locked D5 train-only selection artifact.
Until D5 is reviewed, its FLY/SRQ Ridge lambda remains deliberately unset.

CUB is locked to:

- dataset identity SHA-256:
  `e374af9b576cb6b3503198ef3ea30fd0aa9d2e18c230ff8064e21d4f644af2ca`;
- class mapping SHA-256:
  `caf25ffd97632fcd7dc306425ce88cb5717e653e4d73ccab92c242c0c60ebf83`;
- train content manifest SHA-256:
  `7bddb46ccf7575d0e9ab8976c76a66a58c80dbb09d425bd64aaf5d16e1486f30`;
- test content manifest SHA-256:
  `465f45136552a2e57d9d7cc1a35b5b0ad7148ebebe6e85a502193695c6ca4b19`;
- 5,994 training and 5,794 test images.

The local processed ImageNet-R artifact was audited without image decoding or
feature extraction:

- dataset identity SHA-256:
  `3f3d963b2b0c245ceabc0166c8b1c64d624c2ea31df07ee6ffdbf4cab5f7739d`;
- class mapping SHA-256:
  `dd62e9413ab14ffe4f41035d8dc0c9ba29f07d7c2463bed436a1d1056e6ad385`;
- train content manifest SHA-256:
  `98e87cec5777a9b84fc16a27c2f700c9126bcd5872b88d385ef1fc08c7a2b0c5`;
- test content manifest SHA-256:
  `c24d414ffb410e77943100f249a7b24f7a5c48917b63831b8cb27c1747356609`;
- 23,918 training and 6,082 test files over the same 200 classes.

The audit found 19 content-identical hashes crossing train/test, of which 18
occur under conflicting class directories. It also found 38 within-train and
two within-test duplicate-content groups. The audit report exited with
`FAIL_CROSS_SPLIT_DUPLICATES`; no test image was decoded and no test feature
was created. By project decision, the unchanged artifact may be evaluated
only as a **legacy processed-split** result. Every table and claim must disclose
the overlap; it must not be described as content-disjoint or as an untouched
held-out benchmark.

## Locked methods and hyperparameters

| Method | CIFAR-100 | CUB | ImageNet-R | Role |
|---|---:|---:|---:|---|
| exact FLY-10000 | pending D5 | `1e5` | `1e6` | uncompressed representation control |
| SRQ-FLY-10000 | pending D5 | `1e5` | `1e6` | proposed method |
| exact state-matched FLY | pending D5 (`m=4409`) | `1e5` (`m=4518`) | `1e6` (`m=4518`) | persistent-state-matched control |
| float64 raw Ridge | `0.01` | `100` | `0.01` | raw-feature analytic baseline |

ImageNet-R's matched-width `1e6` value was independently reselected by D2.1
using only inner training. CUB's FLY/SRQ `1e5` values come from D3; raw-Ridge
`100` comes from D4's train-only inner selection. No setting may change after
test extraction begins.

The local replay SOHO method is not a primary exemplar-free comparator because
its learner state retains historical features. If reported, it must appear in
a separately labeled replay-enabled table with sample-level bytes. The
external FLY paper number is context only; the primary comparison uses the
audited local exact-FLY implementation under the shared contract.

## Endpoints and analysis

Primary endpoints per dataset and seed are:

1. final class-incremental accuracy;
2. average incremental accuracy across all 20 stages;
3. persistent learner tensor bytes after every stage;
4. paired SRQ minus exact-FLY-4518 accuracy at matched state.

Secondary diagnostics are stage accuracy, forgetting when valid, prediction
agreement with exact FLY-10000, solver relative residual, update/inference
time, peak runtime memory, serialized checkpoint bytes, and feature/WTA cache
disk bytes. Runtime is descriptive unless hardware/software are identical.

Report every seed plus mean, sample standard deviation, and two-sided 95% t
intervals of paired differences. The same held-out examples recur across
seeds, so intervals describe algorithmic seed/order variation, not population
sampling uncertainty. Do not claim significance unless assumptions are
defended.

## State versus infrastructure

- **Persistent learner state:** sparse projection, compressed factor or exact
  Gram, `Q`, counts, class mapping, classifier, and bounded metadata.
- **Runtime memory:** decoded factors, dense systems, batches, solver
  workspaces, and framework allocations.
- **Experiment cache:** per-sample frozen features, WTA codes, labels, and
  indices on disk. This is not learner state and must not be shipped in an
  exemplar-free checkpoint.

All three byte categories must be reported separately.

## Single-use and integrity rules

Before authorization, a dedicated config, runner, and notebook must be
committed cleanly and their SHA-256 identities inserted here. The runner must:

1. verify repository, config, checkpoint, dataset, class mapping, and content
   identities before loading a test example;
2. fail closed on mismatches, non-finite tensors, backbone key mismatch,
   historical-sample learner tensors, or solver residual above `1e-5`;
3. write append-only provenance before test feature extraction;
4. execute all locked methods/seeds without an accuracy-based early exit;
5. export stage metrics, state inventories, environment, exact commands,
   logs, and hashes;
6. prohibit selection, reranking, or edits using test metrics.

After the first successful extraction of a held-out feature, the protocol is
consumed. Infrastructure failures may be rerun only from identical immutable
code/config after recording the traceback. A numerical or method failure is a
result, not permission to tune.

## Remaining authorization blockers

- CIFAR D5 train-only selection has not yet produced a reviewed real artifact.
- ImageNet-R is permitted only as a disclosed legacy processed-split
  evaluation; it cannot support a content-disjoint held-out claim.
- No dedicated held-out config, runner, or notebook exists.
- Implementation and artifact identities have not undergone independent
  review against the intended checkpoint/state contract.
- No review decision authorizes a single-use test run.

Until every blocker closes in a later reviewed commit, **do not create or open
CIFAR/CUB/ImageNet-R test feature caches and do not run held-out evaluation**.
