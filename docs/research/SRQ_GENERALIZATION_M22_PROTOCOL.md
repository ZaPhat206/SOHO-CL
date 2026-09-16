# M22 — Random / Static / Adaptive selection controls (train-only)

## Scientific question

SRQ-Adaptive stores selected strict-upper factor blocks in FP16 and the rest in
INT8.  Existing results compare it with all-INT8 storage, so two interventions
change together: the FP16 byte allowance and the rule used to choose blocks.
M22 asks whether the factor-MSE selector itself contributes beyond receiving
the same byte budget.

This study is a mechanism ablation.  It uses only a validation split created
from CIFAR-100 training data.  It must not create, contain or read `test.pt`,
and no outcome may change the locked test configuration in the paper.

## Locked comparison contract

- RanPAC phase-2 random Gaussian expansion followed by ReLU, without PETL.
- Frozen ViT-B/16 features from checkpoint SHA-256 `32aa17d6...`.
- Width 20,000, Ridge coefficient `1e6`, ten tasks.
- Three paired streams `s2025`, `s2026`, `s2027`; within a stream every arm
  shares the feature cache, train/validation membership, class order,
  projection, task split and numerical settings.
- Block size 256, INT8 group size 64, QR panel size 128, FP32 diagonal and a
  `uint8` precision flag per upper block.
- Every arm, including Exact and the locked Adaptive implementation, is run in
  this study.  Historical M20 accuracy values are not substituted for a unit.

The main comparison budgets are `beta = {0.01, 0.02, 0.05}` of the byte
difference between an all-INT8 and an all-FP16 strict-upper triangle.  Locked
Adaptive is additionally run at `beta = {0.10, 0.25}` to connect the new curve
to M20 without mixing artifacts.

## Arms

1. **Exact:** FP32 Gram reference.
2. **P2B INT8:** all-INT8 strict-upper factor reference (`beta=0`).
3. **Adaptive:** the existing factor-MSE benefit-per-added-byte selector,
   recomputed after every task.
4. **Static:** factor-MSE selection at task 1; the precision-mask positions are
   then fixed.  Factor values are still updated and re-encoded after every
   task.
5. **Random:** visit block indices in a pseudorandom permutation and greedily
   accept blocks that fit the same byte ceiling.  Two allocation seeds,
   `202501` and `202502`, are registered independently of all data/projection
   seeds.  Random selection does not inspect factor values.

All policies use the same block costs and byte-ceiling formula.  Because edge
blocks have different sizes and the selector is greedy, actual used bytes may
differ below the common ceiling.  Therefore M22 reports both the ceiling and
the actual payload; it does not falsely claim exact byte identity.

## Measurements

For every task and arm:

- validation accuracy on all classes seen so far and AIA;
- relative classifier-weight error and relative logit Frobenius error against
  Exact on the same validation samples;
- prediction agreement with Exact;
- local factor reconstruction error;
- total persistent state, factor state, byte ceiling, actual added bytes,
  selected-block count and precision-mask hash;
- analytic update time and solver residual.

Factor/logit/accuracy observations are outputs only.  They never select a
policy, seed, budget, stopping point or reported run.

## Structural gates and interpretation

There is no accuracy gate.  A complete artifact passes only if:

1. all 48 registered units complete;
2. the locked `s2025` split/projection identity matches M6, and Exact/P2B
   reproduce its width-20k reference within the declared tolerance;
3. every mixed-precision record stays below its byte ceiling;
4. Adaptive, Static and both Random draws have the same ceiling for a given
   stream, task and comparison budget;
5. every Static mask after task 1 equals its task-1 mask;
6. the two Random draws differ in at least one mask for every stream/budget;
7. every solver residual is at most `2e-5`; and
8. every unit records a Tesla T4.

The outcome must be reported whichever policy wins.  Adaptive can be credited
with selection benefit only where it improves an output metric over the
distribution of same-budget Random controls.  Equality supports only the
weaker conclusion that a small mixed-precision allowance is sufficient.

Each unit writes an atomic JSON checkpoint and can be resumed only when its
source/config identity matches.  The exported ZIP excludes Exact weight caches
and all sample-level feature caches.
