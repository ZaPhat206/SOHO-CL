# SRQ-FLY D0 protocol

## Phase Q0: synthetic correctness

Before any ImageNet-R run, synthetic tests must establish deterministic
groupwise quantization, exact diagonals, symmetric Gram reconstruction,
triangular square-root reconstruction, structural SPD, solve residuals,
streaming checkpoint continuation, transactional update failure, state-byte
accounting, task-ID-free logits, and absence of sample-level learner state.

The full repository test suite must also pass. Q0 carries no accuracy claim.

## Phase D0: five-task ImageNet-R training validation

D0 uses training embeddings only. The runner fails closed if `test.pt` is
visible. Hyperparameters and gates are read from the strict config
`configs/srq_fly_imagenetr_d0_train_only.json` and must not be edited after the
run begins.

The runner creates or verifies two WTA experiment caches because exact
FLY-10000 and exact FLY-4096 use different fixed projections. Each cache is
bound to the train-feature SHA-256, representation, seed, stored projection,
and deterministic projection probe. Neither cache may be included in a learner
checkpoint.

Every method unit is resumable and records stage validation accuracy, final
validation average accuracy, state bytes, solver residuals, update time, and
implementation/cache provenance. A compact progress line is printed for every
task. D0 ends after five tasks and never evaluates held-out images or features.

## Interpretation

- If exact FLY-4096 Pareto-dominates SRQ-FLY, prefer the simpler dimension
  reduction and close SRQ-FLY.
- If direct int8 Gram matches FLY and is no less efficient than SRQ-FLY, the
  square-root mechanism is unnecessary and must not be claimed as the cause.
- If float16 square-root fails, int8 square-root is not pursued.
- Only a D0 pass authorizes designing a separately locked 20-task train-only
  study. It does not authorize held-out evaluation.
