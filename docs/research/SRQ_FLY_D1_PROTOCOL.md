# SRQ-FLY D1 locked protocol

Status: implementation contract for a 20-task ImageNet-R training-validation
study. It does not authorize held-out evaluation.

## Authorization from D0

The five-task D0 artifact at commit `a05b749` passed every locked gate. SRQ-FLY
reached validation AA 85.8506% versus 85.8654% for exact FLY-10000 while using
93,166,028 versus 440,006,340 persistent tensor bytes. This result authorizes a
long-horizon train-only drift study, not a test-set run or paper claim.

## Locked design

D1 preserves the D0 checkpoint, preprocessing, training cache, class order,
seed `2025`, 20% deterministic stratified validation split, fixed Ridge
`lambda=1e6`, representations, block/group sizes, and all six methods. The only
scope change is from the first five tasks to all 20 tasks.

The runner evaluates exact FLY-10000 and SRQ-FLY in a paired unit. At every
stage and over every seen-task validation partition it records, without saving
per-sample predictions:

- accuracy for both classifiers;
- prediction agreement;
- relative Frobenius logit error;
- exact and approximate solve residuals;
- persistent tensor bytes.

Exact FLY-4096, raw Ridge, direct-int8 Gram, and float16 square-root FLY remain
mandatory controls. Sample-level feature and WTA caches remain experiment
infrastructure and are not learner state.

## Gates

D1 passes for review only when:

- every method completes all 20 tasks and `test.pt` remains absent;
- all measured solve residuals are at most `1e-5`;
- SRQ-FLY is within 0.50 percentage point of exact FLY-10000 in both average
  incremental validation accuracy and task-20 seen-class validation accuracy;
- minimum paired prediction agreement across stages is at least 98%;
- final SRQ-FLY state is at most 25% of exact FLY-10000;
- SRQ-FLY gains at least 0.10 point over direct-int8 Gram, supporting the
  square-root mechanism rather than quantization alone;
- float16 square-root FLY remains within 0.10 point of exact FLY-10000;
- exact FLY-4096 does not Pareto-dominate SRQ-FLY.

A failed accuracy/mechanism/Pareto gate closes the current SRQ-FLY
configuration. It does not authorize tuning on validation after observing D1,
changing seed, or opening held-out data. A pass authorizes a separately locked
multi-seed train-only protocol only.
