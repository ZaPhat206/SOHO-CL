# Schur Residual SOHO specification

Schur Residual SOHO is the post-confusion hypothesis. It retains CRT-SOHO's
fixed nonlinear sparse/WTA anchor and exact dual-view sufficient statistics,
but selects a low-rank residual subspace from the supervised correction left
after analytically eliminating the anchor classifier. It makes no
confusion-graph novelty claim.

## Full residual system

Let the full raw residual be

```text
C_x = solve(G_pp + eta I, H_px)       # (M,D)
R_x = X - Phi C_x                     # hypothetical (N,D), never stored
```

Its exact sufficient statistics are reconstructed using the equations in
`CRT_SOHO_SPEC.md` with directions `A=I_D`. Define the full block system

```text
A_p = G_pp + lambda_p I               # (M,M)
B   = G_pr                             # (M,D)
D_r = G_rr + lambda_r I               # (D,D)
```

and right-hand side `(Q_p, Q_r)`. Eliminating the anchor block gives

```text
S = D_r - B^T solve(A_p, B)            # (D,D), SPD
T = Q_r - B^T solve(A_p, Q_p)          # (D,C_seen)
W_r,full = solve(S, T).                # (D,C_seen)
```

All operations use Cholesky/triangular solves; no explicit inverse is
permitted.

## Reduced-rank correction

Let `S=L L^T` be the Cholesky factorization and compute

```text
U, sigma, V^T = svd(solve(L, T), full_matrices=False).
```

The best rank-`r` approximation of the full residual coefficient in the
Schur/Ridge objective has coefficient subspace

```text
span(solve(L^T, U[:, :r])).
```

Use Euclidean QR to produce `A_r:(D,r)` with `A_r^T A_r=I`. Reconstruct the
statistics of `R_x A_r` exactly and solve the original anchor/residual block
system. QR changes only the selected subspace and ensures the residual
coefficient penalty remains isotropic and basis-independent.

The effective rank is

```text
min(requested_rank, D, C_seen),
```

because the supervised correction matrix has at most `C_seen` columns. The
runner must report requested and effective rank separately.

## Optimality statement and limits

For fixed sufficient statistics and fixed positive `eta`, `lambda_p`, and
`lambda_r`, truncated SVD above minimizes the full block Ridge objective over
rank-`r` residual coefficient subspaces. With all nonzero correction singular
directions retained, its logits equal the full raw-residual block solution up
to numerical tolerance.

This is conditional optimality, not a generalization guarantee. It does not
prove improvement over raw Ridge, FLY, SOHO, or standard reduced-rank
regression. Direction selection and all Ridge values must be derived from
training data only.

## Persistent learner state

The state remains identical in type to CRT-SOHO:

```text
fixed anchor projection: (M,D), sparse
G_pp: (M,M)
G_xx: (D,D)
H_px: (M,D)
Q_p:  (M,C_seen)
Q_x:  (D,C_seen)
counts/class mapping: (C_seen,)
current A_r: (D,r)
current complement: (M,r)
current classifiers: (M,C_seen), (r,C_seen)
```

No historical image, raw feature row, anchor feature row, label vector, or
sample index is allowed in learner state/checkpoints. The train-validation gate
cache is experiment infrastructure and is not learner state.

## Falsifiable validation gates

1. Numerical relative residual is at most `1e-4` for every candidate.
2. Full residual improves the locked anchor by at least `0.10` percentage
   points on train-only validation.
3. A strict low-rank Schur correction is within `0.50` points of full residual.
4. Schur correction beats the strongest independently selected random,
   standard-Fisher, confusion-Fisher, shuffled-confusion, and
   no-residualization control by at least `0.10` points.
5. Raw Ridge is reported on the identical validation split.
6. Held-out testing is forbidden unless all gates pass and the result is
   reviewed. A paper claim additionally requires multiple class orders,
   backbones/datasets, paired uncertainty, and matched FLY/SOHO comparisons.
