# SRQ-FLY Priority-1 protocol

## Objective

This phase addresses the closest paper blocker without changing the frozen
ViT, sparse projection, WTA code, classifier, or inference rule:

1. remove avoidable SRQ update overhead;
2. measure real PyTorch CUDA peak allocated and reserved memory;
3. compare the six required train-only controls on one paired CIFAR-100
   development stream;
4. defer packed int4 and error feedback until this phase passes.

No command in this phase loads `test.pt`.

## Update optimization

The historical `gram_cholesky` backend remains the checkpoint-compatible
reference. The new opt-in `gram_cholesky_direct` backend exploits the fact that

\[
A_t = Z_t^\top Z_t + \widetilde R_{t-1}^\top\widetilde R_{t-1}
\]

is symmetric in exact arithmetic. PyTorch Cholesky consumes one triangle, so
the direct backend passes `A_t` directly instead of allocating
`(A_t + A_t.T) / 2`, avoiding one dense `m x m` temporary. It is not trusted by
assertion: the isolated benchmark must show predictor drift below the locked
tolerance and unchanged persistent state.

The benchmark also includes the vectorized, checkpoint-compatible compressor
and compares pure analytic-update time against both historical SRQ and dense
Exact FLY. At `m=10,000`, the locked gate is:

- optimized SRQ / Exact FLY update time <= 1.5;
- optimized direct predictor drift <= `1e-5`;
- solver residual <= `1e-5`;
- SRQ persistent bytes unchanged.

Failure stops the dataset ablation. It must not be hidden by averaging feature
extraction or validation time into the update metric.

## Memory measurement

Every method runs in a fresh Python process. After CUDA initialization, the
worker calls `torch.cuda.reset_peak_memory_stats()` and reports:

- absolute peak allocated bytes;
- absolute peak reserved bytes;
- baseline allocated/reserved bytes;
- persistent learner tensor bytes after the final task;
- serialized checkpoint bytes from a temporary `torch.save`.

The synthetic system benchmark measures the analytic update only. The
CIFAR-100 ablation reports the broader method runtime peak including learner
construction, analytic update, and validation, but excludes frozen feature
extraction. These values are not interchangeable with disk feature-cache
bytes or the whole-process/NVML peak.

## Six-way train-only ablation

All methods share seed 2025, the same training/validation split, task order,
ViT features, and paired WTA projection where dimensions agree:

1. `exact_fly_10000`: dense float32 Gram, width 10,000;
2. `srq_int8_optimized`: factor-space groupwise int8, width 10,000;
3. `sqrt_float16`: float16 strict-upper factor, width 10,000;
4. `direct_int8_gram`: direct groupwise-int8 Gram control, width 10,000;
5. `state_matched_exact_fly`: dense Exact FLY at width 4,409, the closest
   non-exceeding state match to SRQ for CIFAR-100;
6. `raw_ridge`: analytic Ridge on the frozen 768D features.

The large FLY-family lambda (`1e6`) and raw Ridge lambda (`0.01`) are inherited
from prior train-only selection. This phase is an ablation, not a new test-set
search. Feature and WTA caches are experiment infrastructure and are excluded
from learner checkpoints.

## Stop rule

Proceed to wider datasets only if the system benchmark and CIFAR train-only
gates pass. True packed int4 must store two nibbles per byte and receive its own
kernel/state audit. Error feedback is a separate method whose residual tensor
must count toward persistent state; neither is implemented in Priority 1.
