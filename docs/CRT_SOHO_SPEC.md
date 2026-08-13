# CRT-SOHO specification

CRT-SOHO (Confusion-aware Residual Transport SOHO) is the proposed successor
to SOHO. It preserves a fixed nonlinear sparse anchor and adds a dynamic
low-rank linear residual branch whose complete historical training statistics
can be reconstructed without replay.

## Scope and non-negotiable invariants

- The backbone and anchor map `phi(x)` are frozen and time-independent.
- Learner state contains no image, sample index, per-example feature, label
  history, replay buffer, or pseudo-example.
- Inference scores all seen global classes without a Task-ID.
- All updates and classifier fits use linear solves, never explicit inverses.
- Hyperparameters are selected from training data only. Test features are
  opened only after a configuration is locked.
- Current FlyCL and SOHO implementations are reference baselines and are not
  modified by CRT-SOHO.

## Fixed anchor and dual-view sufficient statistics

For frozen features `X:(N,D)`, fixed sparse/WTA features
`Phi=phi(X):(N,M)`, and one-hot labels `Y:(N,C)`, retain only

```text
G_pp = Phi^T Phi        # (M,M)
G_xx = X^T X            # (D,D)
H_px = Phi^T X          # (M,D)
Q_p  = Phi^T Y          # (M,C)
Q_x  = X^T Y            # (D,C)
n     = Y^T 1           # (C,)
```

Class columns are maintained in increasing global class-ID order. The fixed
anchor projection is permitted model state; it is not sample-level state.

## Anchor confusion and residual directions

Fit the anchor classifier

```text
W_p0 = solve(G_pp + lambda_p I, Q_p).
```

Let `mu_p,c=Q_p[:,c]/n_c`. Its directed expected pairwise margin is

```text
m_ij = mu_p,i^T (w_p0,i - w_p0,j).
```

For every row `i`, mask its diagonal and compute a relative confusion
distribution

```text
b_ij = softmax_j(-m_ij / tau), j != i
A = (B + B^T) / 2, diag(A)=0.
```

This relative construction avoids absolute Gaussian-tail saturation. The raw
means `mu_x,c=Q_x[:,c]/n_c` and exact within scatter are

```text
S_w = G_xx - Q_x diag(1/n) Q_x^T
S_b^conf = sum_(i<j) A_ij n_i n_j/(n_i+n_j)
            (mu_x,i-mu_x,j)(mu_x,i-mu_x,j)^T.
```

The top `r` generalized eigenvectors of

```text
S_b^conf v = eta (S_w/(N-C) + epsilon I) v
```

form `A_t:(D,r)`. Standard-Fisher, random, shuffled-confusion, and full-raw
directions are mandatory controls.

## Complementary residual branch

Predict the selected raw projection from the fixed anchor:

```text
K = H_px A_t                                            # (M,r)
C_t = solve(G_pp + eta I, K)                            # (M,r)
r_t(x) = x A_t - phi(x) C_t                            # (r,)
```

The no-residualization control sets `C_t=0`. Residualization is intended to
retain information complementary to the anchor rather than duplicate it.

## Exact replay-free reconstruction theorem

For the hypothetical historical residual matrix `R=XA_t-Phi C_t`, define

```text
G_pr = H_px A_t - G_pp C_t
G_rr = A_t^T G_xx A_t
       - (H_px A_t)^T C_t - C_t^T(H_px A_t)
       + C_t^T G_pp C_t
Q_r  = A_t^T Q_x - C_t^T Q_p.
```

Then exactly

```text
G_pr = Phi^T R,   G_rr = R^T R,   Q_r = R^T Y.
```

Thus a newly rebuilt dynamic residual branch has the same block Ridge
statistics as explicit historical reprojection, without retaining any sample.

## Block analytic classifier

Fit

```text
[G_pp + lambda_p I, G_pr                 ] [W_p] = [Q_p]
[G_pr^T,              G_rr + lambda_r I ] [W_r]   [Q_r]
```

with a Cholesky/Schur-complement solve. Inference is

```text
logits(x) = phi(x) W_p + r_t(x) W_r.
```

Raw anchor behavior is recovered by disabling the residual branch. The method
is feature augmentation, not a full-rank change of basis, and therefore is not
covered by the orthogonal no-op result for isotropic Ridge.

## Falsifiable gates

1. `anchor + full raw residual` must improve train-validation accuracy over
   the anchor, otherwise the anchor has no useful missing raw information.
2. A low-rank structured residual must approach the full-raw residual while
   using `r << D`.
3. Confusion residual must beat random, standard-Fisher, and shuffled-confusion
   controls under the same locked policy.
4. Test evaluation is forbidden until gates 1-3 pass on train-only validation.
5. A paper claim additionally requires multiple class-order seeds, datasets,
   backbones, paired uncertainty, and comparison to recent analytic and
   covariance-based exemplar-free methods.
