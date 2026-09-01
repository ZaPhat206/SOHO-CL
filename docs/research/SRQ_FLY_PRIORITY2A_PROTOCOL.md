# SRQ-FLY Priority 2A: allocation-bounded blocked QR

## Scope

Priority 2A addresses the remaining runtime-memory weakness observed in the
immutable Priority-1 artifact.  It does not change the ViT, projection, WTA
code, Ridge value, factor quantizer, persistent tensors, classifier, or
inference rule.  The study is synthetic and cannot access a train or test
cache.

The Priority-1 Tesla T4 result established:

- persistent state reduction: approximately 78%;
- blocked-QR / Exact-FLY update ratio: `1.4706`;
- blocked-QR peak allocated memory remained above Exact FLY;
- predictor drift from the locked SRQ implementation: approximately `5e-11`.

## Candidate update

The existing blocked update applies each panel's Householder reflectors to all
remaining columns in one `torch.ormqr` call.  Both the concatenated right-hand
side and the transformed output coexist.  Priority 2A applies the same
reflectors to fixed trailing-column chunks:

\[
R_{j:j+p,k:k+c},\quad E_{:,k:k+c},
\]

where `p=128` and `c` belongs to the locked grid
`{512,1024,2048,4096}`.  Independent columns receive the same orthogonal
transformation, so chunking changes allocation scheduling rather than the
mathematical QR update.

Experiment runners may also call the explicitly named consuming update after
computing `Z^T Y`.  This permits the disposable dense WTA tensor to serve as the
QR residual buffer.  The ordinary public `update_codes` method remains
non-mutating.

## Repeated isolated benchmark

Every method/repetition runs in a fresh process on one CUDA device.  The
benchmark uses one warm-up round for device-frequency stabilization and seven
measured rounds.  Method order rotates by round.  It compares:

1. dense Exact FLY;
2. the unchunked Priority-1 blocked QR;
3. four chunk sizes fixed above.

The first measured run additionally reports PyTorch allocator peaks for every
update stage.  Feature extraction, dataset caches, and validation are absent.

## Locked selection and gates

For each chunk size, ratios are paired with Exact FLY within the same round.
A candidate is eligible only when all conditions hold:

- maximum relative logit drift from unchunked blocked QR `<=1e-5`;
- persistent tensor bytes exactly unchanged;
- maximum solver residual `<=1e-5`;
- median paired update-time ratio to Exact FLY `<=1.5`;
- median paired peak-allocated ratio to Exact FLY `<=1.05`.

Among eligible candidates, select the smallest median peak ratio, breaking
ties by median update ratio and then smaller chunk size.  If none is eligible,
the phase status is `STOP_MEMORY_GATE`; no threshold may be relaxed after
observing the result.  The next engineering response would be streaming
factor-panel decode/quantize, not dataset evaluation.

## Claims boundary

A pass establishes only allocation/timing feasibility on the recorded CUDA
stack.  It does not establish accuracy, held-out generalization, whole-process
NVML peak, or a universal hardware speed ratio.  A selected candidate must
still pass train-only dataset equivalence before replacing the Priority-1
backend.
