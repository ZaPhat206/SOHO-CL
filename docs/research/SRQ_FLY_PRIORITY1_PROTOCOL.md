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
reference. The allocation-safe `gram_cholesky_direct` control exploits the fact that

\[
A_t = Z_t^\top Z_t + \widetilde R_{t-1}^\top\widetilde R_{t-1}
\]

is symmetric in exact arithmetic. PyTorch Cholesky consumes one triangle, so
the direct backend passes `A_t` directly instead of allocating
`(A_t + A_t.T) / 2`, avoiding one dense `m x m` temporary. It is not trusted by
assertion: the isolated benchmark must show predictor drift below the locked
tolerance and unchanged persistent state.

The final candidate is `blocked_qr`. Given the decoded previous upper factor
and current task code matrix, it computes

\[
R_t = \operatorname{qr}_R\!\left(
\begin{bmatrix}\widetilde R_{t-1}\\ Z_t\end{bmatrix}
\right),
\qquad
R_t^\top R_t = \widetilde R_{t-1}^\top\widetilde R_{t-1}+Z_t^\top Z_t.
\]

Unlike a generic stacked QR, it eliminates 128-column panels using compact
Householder reflectors and applies them only to the panel rows plus the current
rank-update rows. It reuses the decoded factor as its output. This avoids both
the dense `R.T @ R` reconstruction and a new full Cholesky at tasks after the
first, while preserving the same compressed state format. The panel size is
checkpoint-locked and the backend remains ineligible unless its logits pass the
same locked tolerance.

The benchmark also includes the vectorized, checkpoint-compatible compressor
and compares pure analytic-update time against both historical SRQ and dense
Exact FLY. At `m=10,000`, the locked gate is:

- optimized SRQ / Exact FLY update time <= 1.5;
- blocked-QR predictor drift <= `1e-5`;
- solver residual <= `1e-5`;
- SRQ persistent bytes unchanged.

Failure stops the dataset ablation. It must not be hidden by averaging feature
extraction or validation time into the update metric.

### Superseded direct-backend gate

The first T4 system run is retained as a negative engineering result. The
direct backend passed every predictor/state/solver gate and reduced update time
from 1.873 s to 1.310 s, but remained 3.563 times slower than Exact FLY and
therefore failed the locked 1.5 ratio. It also used 1.902 GiB peak allocated
memory versus 1.572 GiB for Exact FLY. The blocked rank-update backend is a
response to that measured bottleneck, not a relaxed gate or a post-hoc accuracy
change.

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

`direct_int8_gram` is a structural negative control, not a required primary
learner. If its quantized symmetric system is not positive definite under the
same locked Ridge value, the worker records `numerical_failure`, failed task,
and reason without adding post-hoc jitter. The remaining five primary methods
must still complete and pass their solver gates. This outcome is reported as
evidence that direct Gram quantization does not preserve SPD; it is never
silently converted into an accuracy result.

## Stop rule

Proceed to wider datasets only if the system benchmark and CIFAR train-only
gates pass. True packed int4 must store two nibbles per byte and receive its own
kernel/state audit. Error feedback is a separate method whose residual tensor
must count toward persistent state; neither is implemented in Priority 1.
