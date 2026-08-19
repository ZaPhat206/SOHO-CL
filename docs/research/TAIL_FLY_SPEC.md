# TAIL-FLY mathematical specification

Status: implementation contract for a falsifiable research prototype. TAIL-FLY
is not yet an established improvement over FLY-CL.

## Scope and motivation

TAIL-FLY (Tail-Aware Incremental Low-rank FLY) keeps the frozen backbone,
fixed sparse FlyHash projection, sample-dependent Top-K/WTA code, and global
analytic classifier of FLY-CL. It changes only how the post-WTA Gram matrix is
stored and solved. This placement is deliberate: a dynamic Top-K map cannot in
general be transported exactly from raw-feature moments by one shared linear
map.

The design combines the continual truncated-SVD recurrence studied by
[LoRanPAC](https://openreview.net/forum?id=bqv7M0wc4x) with an exact coordinate
second-moment tail. It is applied to FLY's sparse WTA representation rather
than LoRanPAC's dense random-ReLU representation. It must therefore be
evaluated as a new hypothesis, not described as LoRanPAC itself.

## Data flow and tensor shapes

At task `t`, for a batch of `b` examples:

- frozen backbone feature: `X_t` in `R^(b x d)`, with `d=768` for the locked
  ViT;
- fixed sparse projection: `R` in `R^(m x d)`;
- WTA code: `Z_t = TopK(X_t R^T)` in `R^(b x m)`;
- global one-hot labels: `Y_t` in `R^(b x C_seen)`;
- retained right/feature singular vectors: `U_t` in `R^(m x r_t)`;
- retained singular values: `s_t` in `R^(r_t)`;
- exact coordinate second moments: `d_t` in `R^m`;
- exact label cross-statistic: `Q_t` in `R^(m x C_seen)`;
- classifier: `W_t` in `R^(m x C_seen)`;
- logits: `Z W_t` in `R^(b x C_seen)`.

Inference receives no task ID and takes the argmax over all seen classes.

## Streaming update

Class columns of `Q` and the counts vector are expanded when new global class
IDs arrive. The exact statistics are

\[
d_t=d_{t-1}+\operatorname{diag}(Z_t^\top Z_t),\qquad
Q_t=\operatorname{expand}(Q_{t-1})+Z_t^\top Y_t.
\]

Let the previous compressed design factor be
`U_(t-1) diag(s_(t-1))`. Form only the temporary matrix

\[
F_t=[U_{t-1}\operatorname{diag}(s_{t-1}),\;Z_t^\top].
\]

The leading at most `r_max` left singular vectors and singular values of
`F_t` become `U_t,s_t`. An implementation may use the standard QR plus small
core-SVD recurrence. It must not retain `Z_t` after `update` returns.

Define the non-negative diagonal tail

\[
\delta_t=\operatorname{clamp}\left(
d_t-\operatorname{diag}(U_t\operatorname{diag}(s_t^2)U_t^\top),0
\right)
\]

and the approximate Gram matrix

\[
\widetilde G_t=U_t\operatorname{diag}(s_t^2)U_t^\top+
\operatorname{Diag}(\delta_t).
\]

In exact arithmetic, repeated truncated SVD produces a positive-semidefinite
under-approximation of the accumulated Gram matrix, so its diagonal cannot
exceed `d_t`. The clamp is only a floating-point safeguard. The exact diagonal
does **not** imply that the corrected approximation always has smaller
spectral-norm error or better accuracy than plain TSVD.

## Analytic classifier

TAIL-FLY solves

\[
(\widetilde G_t+\lambda I)W_t=Q_t,\qquad \lambda>0.
\]

Let `D = Diag(delta + lambda)` and retain only strictly positive singular
values. Woodbury gives

\[
W=D^{-1}Q-D^{-1}U
\left(\operatorname{Diag}(s^{-2})+U^\top D^{-1}U\right)^{-1}
U^\top D^{-1}Q.
\]

The implementation uses Cholesky/linear solves and never forms an explicit
matrix inverse. It reports the relative residual of the approximate system.

Matched controls use the same backbone, projection, WTA code, class order,
split, seed, and evaluation loop:

- exact FLY Ridge with full `G`;
- exact raw-feature Ridge;
- plain truncated-SVD FLY, which drops the diagonal tail;
- diagonal-only FLY, which drops the low-rank term;
- TAIL-FLY.

The plain TSVD control is LoRanPAC-inspired but is not called an official
LoRanPAC reproduction because the nonlinear representation and classifier
protocol differ.

## Persistent learner state

Allowed after a task:

- fixed sparse projection and its configuration;
- `U`, `s`, exact `d`, exact `Q`, class IDs, counts, and scalar metadata;
- optionally the derived classifier `W` for immediate inference.

Forbidden:

- raw images;
- backbone features for historical examples;
- WTA rows, labels, indices, or RNG state indexed by historical sample count;
- replay buffers or disk caches inside a checkpoint.

The serialized checkpoint omits derived `W` and rebuilds it from aggregate
state. Experiment feature/WTA caches are disk infrastructure and must be
reported separately from runtime memory and persistent learner state.

Aggregate state is `O(m r + m C + m)` plus the fixed sparse projection,
instead of exact FLY's `O(m^2 + m C)` Gram state.

## Invariants and mathematical checks

1. `U.T @ U = I` within declared tolerance and `s >= 0`.
2. `d`, `Q`, and counts equal batch oracles independently of SVD truncation.
3. `diag(G_tilde) = d` within tolerance when the unclamped tail is
   non-negative.
4. With no truncation of the accumulated row span, `G_tilde = Z.T @ Z` and
   logits equal exact FLY Ridge within `1e-5`.
5. With rank zero, the solver equals diagonal-only Ridge.
6. The approximate-system relative residual is at most `1e-5` in float64.
7. Checkpoint round-trip preserves logits and contains no tensor dimension
   equal to the historical sample count.
8. `predict_logits` has no `task_id` argument.

For exact Ridge `W=(G+lambda I)^-1 Q` and approximate
`W_hat=(G_tilde+lambda I)^-1 Q`, the resolvent identity gives

\[
\|W-W_{hat}\|_F \le
\frac{\|G-G_{tilde}\|_2\,\|Q\|_F}{\lambda^2}
\]

when both Gram matrices are positive semidefinite and `lambda>0`. For a test
code `z`, logit error is at most `||z||_2 ||W-W_hat||_2`; an argmax is
preserved when the exact winning margin exceeds twice the maximum per-class
logit perturbation. These are conditional stability bounds, not accuracy
guarantees.

## Falsifiable hypotheses

On a locked train-only development split:

- H1: diagonal-tail correction improves validation AA by at least `0.20`
  percentage point over plain TSVD at the same rank and Ridge value;
- H2: TAIL-FLY is within `0.50` point of matched exact FLY;
- H3: TAIL-FLY does not underperform matched raw Ridge;
- H4: resident learner state is at most `25%` of matched exact FLY state;
- H5: every numerical/state invariant above passes.

Failing H1 rejects the tail contribution. Failing H2 or H3 rejects the method
as the next accuracy-preserving FLY replacement at that memory budget.
Passing the development gate only authorizes a separately reviewed held-out
protocol; it is not evidence of generalization or paper readiness.
