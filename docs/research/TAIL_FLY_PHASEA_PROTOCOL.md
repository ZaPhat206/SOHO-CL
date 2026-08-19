# TAIL-FLY Phase A protocol

Status: locked implementation and train-only development plan. No held-out
test evaluation is authorized by this document.

## Phase A0: mathematical correctness

Use synthetic WTA-like matrices only. Required tests:

1. streaming exact `d`, `Q`, and counts equal batch construction;
2. full-rank streaming SVD reconstructs the batch Gram;
3. truncated factors remain orthonormal and have non-negative diagonal tail;
4. Woodbury logits equal a direct solve of the same low-rank-plus-diagonal
   system within `1e-5`;
5. full-rank TAIL-FLY logits equal exact FLY Ridge within `1e-5`;
6. rank-zero TAIL-FLY equals diagonal-only Ridge;
7. checkpoint round-trip and deterministic replay pass;
8. state audit rejects sample-indexed checkpoint tensors;
9. a toy class-incremental stream expands `Q` and predicts all seen classes
   without task ID.

Gate A0 passes only if all tests pass and the maximum approximate-system
relative residual is at most `1e-5` in float64.

## Phase A1: implementation integration

Add TAIL-FLY as an isolated method selected by its own JSON configuration.
Do not edit original SOHO, FlyCL, or prior negative-result implementations.
The runner must validate config keys, seed, cache identity, hidden-test state,
Git provenance, and output schema before any candidate starts.

The exact FLY, raw Ridge, plain TSVD, diagonal-only, and TAIL-FLY candidates
must reuse one fixed projection and one WTA train cache. That cache is
experiment infrastructure and is never serialized in learner state.

Gate A1 requires targeted tests, the complete repository test suite, a clean
diff review, and exact commands recorded in the implementation report.

## Phase A2: ImageNet-R train-only development

ImageNet-R training data is a **development** dataset because its prior
train-validation results motivated TAIL-FLY. It cannot provide pristine
confirmation. Use:

- frozen `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`;
- checkpoint SHA-256
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`;
- ViT pretrained preprocessing;
- 200 classes, 20 class-incremental tasks;
- seed `2025` as required by repository policy;
- deterministic stratified training-validation split;
- no test feature extraction and no `test.pt` in the cache during selection;
- FLY representation `m=10000`, synaptic degree `300`, coding level `0.3`.
- float32 production statistics on the T4 GPU; the locked residual gate remains
  `1e-5`, while standalone mathematical equivalence tests use float64;

Search only the predeclared TAIL rank and Ridge grid. Exact FLY uses its locked
current-task GCV policy; raw Ridge and compressed controls use their declared
train-only grids. Do not edit the grid after observing results.

Development gates:

| Gate | Threshold |
|---|---:|
| maximum solver relative residual | `<= 1e-5` |
| TAIL gain over same-budget plain TSVD | `>= 0.20 pp` |
| TAIL gap from exact FLY | `<= 0.50 pp` |
| TAIL gap from raw Ridge | `>= 0.00 pp` |
| TAIL resident state / exact FLY resident state | `<= 0.25` |
| held-out test remained hidden | required |

Any accuracy-gate failure stops this direction on the locked development
protocol. Do not change seed or dataset merely to hide a failure. A planned
multi-seed analysis is valid only after a single locked configuration is
selected without looking at held-out test data.

## Confirmation boundary

If Phase A2 passes, design a new protocol on a dataset whose outcomes have not
informed the method, with hyperparameters locked before test access. Candidate
datasets and baselines require a separate review. A development pass does not
authorize a held-out run automatically.

Every report must distinguish:

1. per-sample feature/WTA cache on disk;
2. peak runtime memory;
3. resident learner tensors after a task;
4. serialized checkpoint tensors allowed to survive process restart.
