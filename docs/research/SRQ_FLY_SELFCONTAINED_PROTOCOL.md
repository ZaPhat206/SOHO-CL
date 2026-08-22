# SRQ-FLY self-contained three-dataset protocol

Status: **IMPLEMENTED; REAL COLAB RUN PENDING.** This protocol starts without
using any previously selected Ridge value. It performs train-only selection,
locks the selected values, refits from empty learner state on all official
training examples, and only then evaluates the official test split.

## Research question and methods

The primary comparison is Exact FLY-10,000, SRQ-FLY-10,000, and raw-feature
Ridge. All three use the same frozen ViT checkpoint, preprocessing, dataset
split, task order, and final replicate identities. Exact FLY and SRQ-FLY use
the same width-10,000 sparse projection and WTA codes within a replicate.

The shared FLY-family Ridge value is intentionally selected from the mean
inner-validation AIA of Exact FLY and SRQ-FLY. This keeps the paired comparison
focused on square-root-state compression rather than giving the approximation
a different regularizer. Raw Ridge selects its own value from the identical
numeric grid. This is a declared paired-compression protocol, not a claim that
each FLY-family member received an independently optimized lambda.

## Train-only nested selection

For every class, the official training split is deterministically shuffled
with split seed `2025` and divided as follows:

1. 20% is reserved as outer validation;
2. the remaining 80% is the development partition;
3. 20% of development is inner validation and the remainder is inner fit.

Thus the nominal proportions are 64% inner fit, 16% inner validation, and 20%
outer validation. Per-class rounding is recorded in index hashes. No sample is
shared between partitions. The test feature file must be physically absent
during this phase.

The fixed Ridge grid is:

`1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8`.

Every candidate is evaluated on three development replicates. Exact FLY and
SRQ-FLY use their mean inner-validation AIA as the paired family score; raw
Ridge uses its own inner-validation AIA. Ties select the larger lambda. The
chosen values are evaluated once on outer validation without further search.
If either selected value is a grid endpoint, the protocol stops before test so
the grid cannot be widened after test use.

## Randomness separation

- split seed: `2025`;
- development class-order seeds: `2025, 2026, 2027`;
- development projection seeds: `4201, 4202, 4203`;
- final class-order seeds: `3031` through `3036`;
- final projection seeds: `5031` through `5036`.

Split, class-order, and projection randomness have separate sources. Final
replicates are disjoint from development replicates. Within a final replicate,
all methods receive the same class order; Exact FLY and SRQ-FLY additionally
share the exact projection and WTA cache.

## Final refit and evaluation

After all three `selection.json` files pass their contracts, `lock` binds the
protocol, runner, method sources, selected values, Git commit, and selection
file hashes in `authorization.json`. Test extraction refuses to run without a
matching authorization. Any subsequent protocol, runner, or selection change
invalidates the authorization.

For each final replicate, every learner starts from empty state and processes
all official training features task by task. The frozen backbone is not
fine-tuned. “Refit” therefore means rebuilding only the analytic continual
learner and classifier, not training ViT. Prediction is task-ID-free and covers
all classes seen so far.

The primary outputs are final accuracy, average incremental accuracy,
forgetting, per-task average seen-class accuracy, persistent learner-state
bytes, update time, and inference time. Results include all six replicates,
sample standard deviation, and two-sided 95% t intervals. There is no
accuracy-based test stopping or test-set hyperparameter tuning.

## State and cache boundary

Persistent learner state may contain the fixed sparse projection, exact Gram
or compressed square-root factor, `Q`, counts, classifier weights, and bounded
metadata. It may not contain historical images, features, WTA codes, labels,
or sample indices. Feature and WTA caches are sample-level experiment
infrastructure on `/content`; their disk bytes are reported separately and the
caches are excluded from the evidence ZIP.

## Dataset disclosure

CIFAR-100 and CUB-200-2011 use their official processed train/test splits. The
legacy processed ImageNet-R artifact contains 19 byte-identical hashes across
train and test, including 18 under conflicting class directories. Its result
must always be labeled **legacy processed split, not content-disjoint**.

## Immutable implementation

- protocol: `configs/srq_fly_selfcontained_final.json`;
- runner: `tools/srq_fly_selfcontained.py`;
- notebook: `notebooks/srq_fly_selfcontained_final_colab.ipynb`.

The notebook verifies protocol and runner SHA-256 values before any dataset is
used. Source hashes are updated only before the first real run, never after
observing held-out results.
