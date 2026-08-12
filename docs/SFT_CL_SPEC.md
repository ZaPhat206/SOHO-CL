# SFT-CL specification

SFT-CL (Sufficient-statistic Fisher Transport) is a strict exemplar-free
control/proposal built in a *fixed* frozen-feature space. It is separate from
the legacy SOHO pipeline and does not alter FlyCL or SOHO source semantics.

For frozen features `X:(N,D)` and one-hot seen-class targets `Y:(N,C)`, the
only sample-derived learner state is

```text
G = X^T X                 # (D,D)
Q = X^T Y                 # (D,C)
n = Y^T 1                 # (C,)
```

with global dataset class IDs kept in increasing canonical order. No image,
dataset index, per-example feature, label history, or replay tensor is allowed
in an SFT-CL checkpoint.

## Reconstructed geometry

Class means and exact within scatter are reconstructed from the state:

```text
mu = Q diag(1 / n)                                      # (D,C)
Sw = G - Q diag(1 / n) Q^T                             # (D,D)
Sb = sum_c n_c (mu_c - mu_bar)(mu_c - mu_bar)^T        # (D,D)
```

For confusion methods, first solve raw Ridge

```text
W0 = solve(G + lambda0 I, Q)                            # (D,C)
```

and approximate pairwise class error under a shared pooled covariance
`Sigma=Sw/(N-C)`:

```text
m_ij = mu_i^T (w_i - w_j)
v_ij = (w_i - w_j)^T Sigma (w_i - w_j)
p_i_to_j = Phi(-m_ij / sqrt(v_ij + epsilon))
a_ij = (p_i_to_j + p_j_to_i) / 2
```

The corresponding pairwise Fisher scatter is

```text
Sb_conf = sum_{i<j} a_ij * n_i*n_j/(n_i+n_j)
          * (mu_i-mu_j)(mu_i-mu_j)^T.
```

`shuffled_confusion_fisher_soft` preserves the multiset of `a_ij` weights but
deterministically permutes their class pairs; it is a semantic control.

## Transport and classifier

Let `Sw/(N-1) + eps I = L L^T` and

```text
C = L^-1 (Sb_star / N) L^-T = U diag(eta) U^T.
```

For hard Fisher, use `A=L^-T U_r`. For soft Fisher, retain all `D` directions:

```text
s_i^2 = delta + (1-delta) eta_i/(eta_i+kappa),  0 < delta <= 1
A = L^-T U diag(s)                                  # (D,D)
```

Then reconstruct transformed Ridge statistics without historical features:

```text
Gz = A^T G A
Qz = A^T Q
P  = solve(Gz + lambda I, Qz)
logits(x) = x A P
```

For a hypothetical full historical matrix `X`, these statistics exactly equal
`(XA)^T(XA)` and `(XA)^T Y`. Thus a dynamic linear `A` can be rebuilt at each
task without replay.

For an invertible square `A`, the resulting predictor equals anisotropic Ridge:

```text
A solve(A^T G A + lambda I, A^T Q)
= solve(G + lambda (A A^T)^-1, Q).
```

If `A` is orthogonal this reduces to ordinary isotropic raw Ridge. The proposed
effect therefore requires a non-orthogonal transport; it is not an orthogonal
change-of-basis claim.

## Supported methods

`sft_raw_ridge`, `fisher_hard`, `confusion_fisher_hard`, `fisher_soft`,
`confusion_fisher_soft`, and `shuffled_confusion_fisher_soft` are selected
explicitly through `tools/experiment_runner.py`. Legacy T-SOHO methods retain
their old names and behavior. `cached_flycl` and `cached_soho_replay` are cache
controls; the latter serializes feature replay and must always be reported as
non-exemplar-free.

The template [sft_cl_cifar100_template.json](../configs/sft_cl_cifar100_template.json)
is a method-specific starting configuration, not a locked experimental result.
Its `lambda`, `kappa`, and `delta` values must be replaced only by a train-only
selection result before test evaluation.
