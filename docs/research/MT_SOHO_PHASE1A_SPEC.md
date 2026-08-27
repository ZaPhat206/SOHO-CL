# MT-SOHO Phase 1A: train-only feasibility

## Question

Does a supervised, sample-independent projection placed after a fixed WTA
anchor improve class-incremental accuracy without retaining historical
samples? Phase 1A is a feasibility test, not a held-out result.

## Learner

For frozen backbone features `X_t` and labels `Y_t`, the fixed anchor is

\[
U_t=\operatorname{TopK}(X_tW^\top),
\]

where `W` is a deterministic sparse Gaussian projection created once. The
learner accumulates

\[
G_u\leftarrow G_u+U_t^\top U_t,\quad
Q_u\leftarrow\operatorname{expand}(Q_u)+U_t^\top Y_t,
\]

and raw-view moments `G_x=X^T X`, `Q_x=X^T Y`, and class counts. They recover
class means and pooled within-class scatter exactly:

\[
\mu_c=Q_x[:,c]/n_c,\qquad
S_w=G_x-\sum_c n_c\mu_c\mu_c^\top.
\]

Regularized whitening and truncated SVD produce unit-norm target prototypes
`P_t` of shape `(C_seen, r)`. The post-WTA projection is

\[
B_t=\operatorname{solve}(G_u+\alpha I,Q_uP_t).
\]

For `V=UB_t`, its all-history moments are transported exactly:

\[
G_v=B_t^\top G_uB_t,\qquad Q_v=B_t^\top Q_u.
\]

The prediction combines the fixed anchor Ridge head and the transported head:

\[
\ell(x)=uW_u+\gamma uB_tW_v.
\]

Every linear system uses Cholesky solves. No explicit inverse is permitted.

## Persistent state

Allowed tensors are fixed projection metadata, `G_u`, `Q_u`, `G_x`, `Q_x`,
the cross-view moment, counts, current heads, `B_t`, and current target
prototypes. No tensor may contain a historical sample-count dimension. No
image, label vector, frozen feature, WTA code, pseudo-sample, or replay index is
allowed in a checkpoint.

## Claims and limitations

1. Exact transport means equality to batch recomputation **in the fixed WTA
   space for the current sample-independent `B_t`**. It is not equivalence to
   legacy SOHO's dynamic pre-WTA map.
2. A linear post-WTA projection does not add information to `U`; a gain can
   only come from supervised metric shaping and anisotropic regularization.
3. Phase 1A uses width 1,000 to test mechanism feasibility. It cannot establish
   performance at FLY's reported width 10,000.
4. The feature cache is experiment infrastructure and may contain per-sample
   training features. It is not learner state and must not be serialized with
   the learner.
5. SRQ is deliberately excluded until the uncompressed accuracy gate passes.

## Train-only protocol

- dataset: CIFAR-100 training split only;
- frozen feature identity: ViT-B/16 checkpoint and preprocessing already
  locked in the repository;
- split: deterministic class-stratified nested holdout;
- seeds: three predeclared class-order/projection pairs, with protocol seed
  `2025`;
- inner validation selects anchor Ridge and the MT target grid;
- outer validation evaluates the locked candidate and controls;
- `test.pt` being visible is a hard failure.

Controls are fixed-WTA Ridge, unwhitened targets, and shuffled class targets.
The fixed anchor Ridge grid is `{1e2,1e4,1e6,1e8}`, covering the scale used by
the original FLY implementation. After it is locked, the projection reuses
that coefficient; the low-dimensional adapted head separately searches
`{1,100}`. Rank, covariance shrinkage, and adaptation weight each use two
predeclared values, yielding 16 MT candidates.

The branch advances only if all of the following hold:

- solver relative residual at most `1e-4`;
- every method is exemplar-free;
- MT-SOHO exceeds fixed-WTA by at least `0.20` pp outer-validation AIA;
- MT-SOHO exceeds shuffled targets by at least `0.10` pp;
- whitening exceeds unwhitened targets by at least `0.05` pp.

Passing authorizes a matched-width comparison with legacy SOHO replay. It does
not authorize test evaluation or a 10,000-dimensional study.
