# SRQ-FLY D0 specification

Status: implementation contract for a five-task, training-validation diagnostic.
It is not a held-out experiment and does not authorize a paper claim.

## Question

The previous CertiFLY Q1 run stopped because a cumulative Frobenius error bound
could not certify direct block-quantized Gram matrices. D0 separates three
possibilities:

1. direct int8 quantization is accurate and only the bound was too loose;
2. a positive-definite square-root state is more accurate or stable;
3. reducing the ordinary FLY expansion dimension gives a stronger
   accuracy-memory frontier than either quantized method.

## Locked representation and stream

All methods use the same frozen ViT features, preprocessing, class order, seed
`2025`, deterministic 20% training-validation split, and first five tasks of
the locked 200-class ImageNet-R stream. Held-out features must remain absent.

The principal representation is the existing signed largest-value FLY/WTA code
with `m=10000`, synaptic degree `300`, and coding level `0.3`. The compact-FLY
control changes only `m` to `4096`. D0 fixes `lambda=1e6`, the value selected by
current-task GCV at all 20 stages of the preceding matched-FLY train-only run.

## Direct groupwise-int8 Gram

The direct control updates

\[
\widetilde G_t=Q_8(\operatorname{decode}(\widetilde G_{t-1})+Z_t^\top Z_t).
\]

It stores the diagonal exactly in float32. Strict-upper entries are grouped in
deterministic blocks of 64 and symmetrically quantized to int8 with one float32
scale per group. Reconstruction mirrors the upper triangle exactly. The Ridge
system is accepted only when Cholesky succeeds and its measured relative
residual is within the locked numerical gate. No worst-case certificate is
used as a stopping condition.

## Square-root state

For fixed Ridge `lambda`, define

\[
A_t=G_t+\lambda I=R_t^\top R_t.
\]

The streaming square-root update is semantically

\[
\bar R_t=\operatorname{qr}\begin{bmatrix}
\operatorname{decode}(\widetilde R_{t-1})\\ Z_t
\end{bmatrix},\qquad
\widetilde R_t=Q(\bar R_t).
\]

The implementation may equivalently form
`R_prev.T @ R_prev + Z_t.T @ Z_t` and apply Cholesky. It must never form an
explicit inverse. Two storage controls are locked:

- `sqrt_float16`: exact float32 diagonal and float16 strict upper triangle;
- `srq_int8`: exact float32 diagonal and groupwise-int8 strict upper triangle.

For either representation, the decoded triangular factor has a positive exact
diagonal, hence

\[
\widehat A_t=\widetilde R_t^\top\widetilde R_t\succ0.
\]

The classifier is obtained from two triangular solves:

\[
\widetilde R_t^\top U_t=Q_t,\qquad
\widetilde R_t W_t=U_t.
\]

This structural SPD guarantee is not an accuracy guarantee. Quantization drift
and factor perturbation are recorded after every task.

## Learner state

Allowed persistent state is the fixed sparse projection, quantized Gram or
triangular factor, exact `Q`, class counts/mapping, derived classifier, and
bounded scalar/group metadata. Images, historical backbone features,
historical WTA rows, sample labels/indices, and any sample-count-shaped replay
tensor are forbidden. Prediction is global and has no task-ID argument.

Feature and WTA caches are sample-level experiment infrastructure on disk, not
learner state, and must be reported separately.

## Mandatory controls and gates

D0 compares exact FLY-10000, exact FLY-4096, raw Ridge, direct groupwise-int8
Gram, float16 square-root FLY, and int8 SRQ-FLY.

D0 passes for review only if:

- all methods complete five tasks without NaN/Inf;
- every solve residual is at most `1e-5`;
- int8 SRQ-FLY is within `0.50` percentage point of exact FLY-10000;
- final SRQ-FLY state is at most 25% of exact FLY-10000;
- SRQ-FLY is not Pareto-dominated by exact FLY-4096;
- held-out `test.pt` remains absent.

Failure stops the direction. It does not authorize changing seed, group size,
task count, or dataset after observing the result.
