# SRQ-FLY P2B final confirmation protocol

## Purpose

This phase estimates the accuracy, learner-state and paired-predictor behavior
of the accepted Priority-2B implementation on the same three datasets and six
replicates used by the earlier self-contained SRQ-FLY study. It is a backend
confirmation. It is not a new hyperparameter search, not an accuracy-improvement
claim and not a fresh first-use held-out evaluation.

Priority-2B is the final backend because it passed the preceding compatibility
and memory phases. Priority-2C is excluded: its implicit-Ridge initialization
reduced real peak allocation but failed the locked real-data predictor and
stage-accuracy equivalence gates.

## Locked method

All FLY-family methods use the frozen ViT-B/16 representation, a width-10,000
sparse projection with synaptic degree 300, coding level 0.3, and the same WTA
codes within each replicate.

The P2B learner stores the upper square-root factor using groupwise int8 blocks,
uses blocked QR updates with panel size 128, initializes the first task through
Gram-Cholesky, and streams quantization in batches of 64 blocks. Exact FLY
stores the dense float32 Gram. Raw Ridge operates on the same frozen 768-D ViT
features and has a separately selected Ridge value.

## Selection and evaluation

The only accepted selection evidence is
`srq_fly_selfcontained_three_dataset_results.zip`, SHA-256
`e4b630781ff6f69deaecb63dda9926d256cd6b654ef4b51a682bf3ef94e6490b`.
The three selection JSON files are verified byte-for-byte. Selected values are:

- CIFAR-100: FLY/SRQ `1e6`, raw Ridge `100`;
- CUB-200-2011: FLY/SRQ `1e5`, raw Ridge `100`;
- ImageNet-R: FLY/SRQ `1e6`, raw Ridge `1000`.

The final class-order seeds are 3031-3036 and projection seeds are 5031-5036,
paired by position. The task protocols remain CIFAR-100/10, CUB/20 and
ImageNet-R/20. No result can change the number of units executed, and no test
metric changes a lambda, method setting or seed.

## Reported quantities

For each dataset and method, report mean, sample standard deviation and 95%
confidence interval across six replicates for final accuracy, average
incremental accuracy, forgetting, persistent tensor bytes, analytic update
time and inference time. Report a paired 95% interval for P2B minus Exact FLY
average incremental accuracy. Also report per-task prediction agreement and
relative logit error between P2B and Exact FLY.

There is deliberately no post-test accuracy gate. The observed effect and its
interval are reported. Persistent tensor bytes are not called peak runtime
memory; isolated peak-memory evidence remains in the Priority-2A/2B artifacts.
Feature and WTA caches are sample-level experimental infrastructure and are
excluded from learner checkpoints and the exported evidence ZIP.

## Required disclosures

All three test splits were previously consumed by the earlier locked SRQ-FLY
run, so results are confirmation evidence rather than untouched held-out
evidence. ImageNet-R is a legacy processed split with 19 duplicate content
hashes across train/test, 18 under conflicting labels, and must not be described
as content-disjoint.
