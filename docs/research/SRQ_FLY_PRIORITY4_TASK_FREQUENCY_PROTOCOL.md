# SRQ-FLY Priority 4: task-frequency robustness protocol

## Scientific question

SRQ-FLY decodes, updates, and re-quantizes its square-root factor after each
task. Does doubling this operation count from 10 to 20 materially increase the
approximation error?

This is a CIFAR-100 **train-only** robustness experiment. It cannot authorize
or evaluate the held-out test split.

## Paired intervention

For each of five preregistered development replicates, the following are fixed
between the 10-task and 20-task schedules:

- the 50,000 frozen ViT-B/16 training features;
- per-class train/validation membership;
- class order;
- sparse projection and WTA code for every training sample;
- width 10,000, synaptic degree 300, coding level 0.3;
- Ridge coefficient \(10^6\);
- Exact FLY and SRQ-P2B source identities.

Only task grouping changes. The 10-task schedule presents 10 classes per
update; the 20-task schedule presents 5. The class order is identical, so
20-task stages 2, 4, ..., 20 correspond to the same seen-class sets as 10-task
stages 1, 2, ..., 10.

Validation membership is generated once per class before task grouping. This
prevents the task schedule from silently changing the data split.

## Metrics

- `aligned_AIA`: mean accuracy at the ten common seen-class checkpoints;
- final validation accuracy;
- SRQ-minus-Exact accuracy gap within each schedule;
- `added_frequency_loss`: `(Exact-SRQ at 20 tasks) - (Exact-SRQ at 10 tasks)`;
- Exact final prediction agreement between schedules, a protocol sanity check;
- SRQ/Exact prediction agreement, solver residual, persistent state, and update
  time.

Ordinary 20-stage AIA is retained as descriptive output but is not compared
directly with 10-stage AIA, because it averages different evaluation
checkpoints.

## Locked hypotheses

Across the five replicates:

- mean aligned SRQ loss relative to Exact is at most 0.50 pp for both schedules;
- mean added-frequency loss is at most 0.25 pp;
- mean 20-task final loss relative to Exact is at most 0.50 pp;
- every solver residual is at most \(2\times10^{-5}\);
- SRQ/Exact prediction agreement is at least 0.98;
- Exact FLY final predictions are schedule-invariant to at least 0.995 and its
  final accuracy differs by at most 0.05 pp (allowing float32 summation-order
  effects from different task grouping);
- SRQ persistent tensor state is at most one quarter of Exact FLY state.

A failed accuracy gate is a scientific result and must not be repaired by
changing seeds, Ridge, representation, or stopping rules.

## Isolation and resumability

Each `(replicate, task schedule)` pair runs in a separate process and writes an
immutable unit JSON. A rerun restores a unit only when its config and source
hashes match. WTA codes are sample-level experiment cache, not learner state,
and are excluded from the exported evidence bundle.

## Interpretation boundary

A pass supports robustness to twice as many factor quantization events on
CIFAR training-validation streams. It is not a new held-out result, does not
show robustness beyond 20 tasks, and does not establish the same behavior for
another backbone, dataset, bit width, or factor update algorithm.
