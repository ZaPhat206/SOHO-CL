# CARS-FLY specification

Status: implementation contract for a train-only research pilot. CARS-FLY does
not replace or modify the existing FLY-CL or SOHO learners.

## Claim boundary

CARS-FLY (Conditional Analytic Residual Sketching for FLY) is an
exemplar-free, task-ID-free analytic learner. Its intended contribution is a
certified accuracy-memory trade-off: a compact, fixed nonlinear FlyHash/WTA
anchor is augmented by the smallest raw-feature correction subspace that
captures a declared fraction of the label-predictive signal left unexplained by
the anchor.

CARS-FLY does not claim that a linear residual creates information outside the
joint view `[phi(x), x]`. It does not transport Top-K, adapt the backbone,
generate pseudo-features, or use a confusion graph to select its directions.

## Fixed views and shapes

For a batch of `N` samples:

- frozen backbone features: `X in R^(N x d)`;
- fixed sparse projection: `W in R^(m x d)`;
- fixed WTA anchor: `Phi = TopK(X W^T) in R^(N x m)`;
- global one-hot targets: `Y in R^(N x C_seen)`.

The projection, Top-K semantics, backbone, and preprocessing never change.

## Persistent sufficient statistics

The learner updates only:

```
G_pp = sum Phi^T Phi       shape (m, m)
G_xx = sum X^T X           shape (d, d)
H_px = sum Phi^T X         shape (m, d)
Q_p  = sum Phi^T Y         shape (m, C_seen)
Q_x  = sum X^T Y           shape (d, C_seen)
n     = class counts       shape (C_seen,)
```

No tensor may have a dimension equal to the historical sample count. Feature
and WTA row caches are experiment infrastructure and are forbidden from learner
checkpoints.

## Conditional correction

Let `lambda_p`, `lambda_r`, and `eta` be positive. First residualize raw
features against the anchor:

```
C_full = solve(G_pp + eta I_m, H_px)
R_full = X - Phi C_full
```

All moments of `R_full` are reconstructed exactly from the retained statistics.
After eliminating the anchor block from the regularized normal equations, define

```
K = G_pp + lambda_p I_m
S = G_rr + lambda_r I_d - G_pr^T solve(K, G_pr)
T = Q_r - G_pr^T solve(K, Q_p)
```

where `G_pr`, `G_rr`, and `Q_r` are the exact reconstructed moments of
`R_full`. With `S = L L^T`, compute

```
M = solve(L, T) = U diag(sigma) V^T.
```

The total regularized conditional correction energy is

```
Delta_full = sum_j sigma_j^2.
```

For a declared `energy_threshold kappa in (0, 1]`, select the smallest
`r <= max_rank` satisfying

```
sum_{j <= r} sigma_j^2 / Delta_full >= kappa.
```

If `Delta_full <= minimum_objective_gain`, select `r=0` and use the anchor
classifier only. If `max_rank` cannot reach `kappa`, select `max_rank` and
report the attained fraction rather than silently changing the budget.

For `r>0`, map the selected Schur-whitened subspace back to raw coordinates and
use an orthonormal basis `A_r in R^(d x r)`. The deployed residual view is

```
C_r = solve(G_pp + eta I_m, H_px A_r)
R_r(x) = x A_r - phi(x) C_r.
```

One global block Ridge classifier is solved on `[phi(x), R_r(x)]` using linear
solves only. Inference never accepts a task identifier.

## Exact backfilling identity

For `u(x) = [phi(x), x]` and any later linear view `z(x) = u(x) M`,

```
G_z = M^T G_u M
Q_z = M^T Q_u.
```

This identity is exact although `phi` contains Top-K, because Top-K is evaluated
once when each sample arrives and its joint moments with `x` are retained. It
does not imply that a new Top-K map can be reconstructed from raw moments.

## Certificates and invariants

1. Streaming `G_pp`, `G_xx`, `H_px`, `Q_p`, and `Q_x` equal their batch forms.
2. Reconstructed residual moments equal moments obtained by materializing the
   residual rows, within the declared numerical tolerance.
3. Selected rank is the smallest permitted rank reaching `kappa`, unless the
   configured maximum rank is insufficient.
4. `captured_energy = sum_{j <= r} sigma_j^2` and
   `tail_energy = sum_{j > r} sigma_j^2` are finite and non-negative.
5. At full effective rank, CARS-FLY reproduces the full raw-residual block Ridge
   predictor up to solver tolerance.
6. Every solve uses Cholesky/triangular solve or another declared linear solver;
   no explicit matrix inverse is permitted.
7. The learner checkpoint contains only configuration, fixed projection, class
   mapping, counts, and aggregate sufficient statistics. Derived directions and
   classifier weights are rebuilt deterministically.
8. Persistent state dimensions depend on `d`, `m`, `C_seen`, and selected rank,
   never on `N_seen`.

## Falsifiable hypotheses

- H1: conditional Schur selection beats random and standard-Fisher residual
  controls under the same anchor, rank budget, and Ridge policy.
- H2: adaptive rank reaches a useful accuracy-memory Pareto point relative to
  matched FLY-CL; it is not required to exceed FLY accuracy to survive.
- H3: the retained-energy certificate predicts the regularized objective gap;
  it is not claimed to predict classification accuracy perfectly.
- H4: exact reconstructed backfilling matches an oracle that materializes all
  rows. Failure falsifies the implementation, not merely the hypothesis.
- H5: if full joint raw+anchor Ridge provides no train-only gain over the
  compact anchor, the residual branch has no useful headroom and the phase
  stops.

## Known limitations

- The method stays inside the linear hypothesis class over `[phi(x), x]` and
  changes regularization/rank rather than adding new information.
- Exact joint moments cost `O(m^2 + md + d^2)` memory. A full `m=10000` anchor
  remains expensive; the primary pilot therefore studies a compact anchor.
- Selecting directions requires solving systems involving the anchor Gram.
- The objective certificate does not guarantee a classification-accuracy gain.
- Confusion geometry is retained only as a falsification control because prior
  repository experiments did not distinguish it from shuffled confusion.
