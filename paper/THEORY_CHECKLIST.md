# SRQ-FLY theory checklist

This file separates proven algebraic properties from empirical hypotheses.
The final paper must not promote a proof sketch or implementation invariant to
a theorem without a complete proof and assumptions.

## T1: exact streaming sufficient statistics

Statement: for exact codes and exact arithmetic,

\[
G_t=\sum_{k=1}^t Z_k^\top Z_k,
\qquad
Q_t=\sum_{k=1}^t Z_k^\top Y_k
\]

equals batch accumulation over the same stream after global class-column
expansion.

- proof: induction over tasks;
- implementation test: already covered by streaming analytic tests;
- caveat: this does not transport dynamic WTA codes from raw feature moments.
- complete proof: `PROOFS.md`, Lemma 1.

## T2: exact square-root recursion

Statement: with `R_0^T R_0=lambda I` and no quantization,

\[
R_t^\top R_t=\lambda I+\sum_{k=1}^t Z_k^\top Z_k.
\]

- proof: induction using the Cholesky update;
- verify equivalence of the implemented lower/upper convention;
- state explicitly that finite-precision and quantized SRQ follow an
  approximate recursion.
- complete proof: `PROOFS.md`, Theorem 1.

## T3: SPD by construction

Statement: if decoded triangular factor `R_t` has strictly positive diagonal,
then `R_t^T R_t` is positive definite.

- proof: `x^T R^T R x = ||Rx||_2^2 > 0` for nonzero `x` because `R` is
  nonsingular;
- implementation invariant: diagonal is unquantized float32 and checked
  positive;
- limitation: SPD does not imply small classifier or prediction error.
- complete proof: `PROOFS.md`, Theorem 2.

## T4: Ridge perturbation bound

For `A_t` and `A_t + Delta_t` positive definite, derive with the resolvent
identity

\[
(A+\Delta)^{-1}-A^{-1}=-A^{-1}\Delta(A+\Delta)^{-1}.
\]

Then bound classifier and per-code logit error. Required additions:

- specify Frobenius versus spectral norms consistently;
- expose dependence on `lambda_min(A_t)` and `lambda_min(A_t+Delta_t)`;
- relate factor error `R_tilde-R_t` to Gram error
  `R_tilde^T R_tilde-R_t^T R_t`;
- do not infer argmax preservation without an explicit classification-margin
  condition.
- completed in `PROOFS.md`, Sections E--F; a stream-level non-accumulation
  bound remains open.

## T5: conditional argmax preservation

Potential corollary: if the exact top-one logit margin for code `z` is greater
than twice a uniform per-class logit perturbation bound, SRQ preserves the
prediction. This is conditional and sample-specific. It does not establish the
preregistered 98% empirical agreement gate.

Completed as a conditional result in `PROOFS.md`, Section G.

## T6: state complexity

Provide exact byte formulas for:

- sparse CSC projection values and indices;
- int8 strict triangle;
- unquantized float32 diagonal;
- one float32 scale per quantization group;
- class cross-statistic, classifier, and counts;
- exact-FLY dense Gram/factor control.

Check formulas against `persistent_tensors()` and runtime values at every task.
Derived classifier bytes must be counted if the checkpoint retains weights.

The implementation formula is recorded in `PROOFS.md`, Section H. Scale
groups restart per stored block, so a single global ceiling is not exact.

## T7: exemplar-free and task-ID-free invariants

The checkpoint must contain no historical sample, image, feature, WTA code,
label, sample index, or sample-count-shaped tensor. The inference signature
must not accept task identity. These are audited software/state properties,
not probabilistic statements.

## Statements explicitly not available

- quantization improves accuracy;
- SRQ exactly matches uncompressed FLY;
- the quantization error cannot accumulate across tasks;
- convergence results for 4-bit Shampoo transfer to streaming Ridge;
- full-dimensional orthogonal transforms add predictive information under
  isotropic Ridge;
- dynamic Top-K is representable by one sample-independent linear transport.
