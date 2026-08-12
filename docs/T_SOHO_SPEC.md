# T-SOHO specification

Status: design contract only. This document specifies the intended Transportable SOHO (T-SOHO) learner; it does not authorize a model implementation.

## Scope

T-SOHO is a class-incremental learner on a frozen pretrained backbone. It must be exemplar-free in learner state, train from streaming sufficient statistics, and predict over all classes seen so far without a Task-ID. Its proposed novelty is a **strict low-rank, confusion-geometry-informed projection in label/logit space**. It is not an orthogonal feature transport method and does not depend on a dynamic Top-K operator.

## Notation and tensor shapes

Let `D` be the frozen backbone feature dimension, `C` the total configured class universe, and `S_t⊆{0,…,C−1}` the classes observed through task `t`; `C_t=|S_t|`.

| Symbol | Shape | Definition |
|---|---:|---|
| `x` | `(B,D)` | frozen feature batch from the backbone; `B` is batch size. |
| `h(x)` | `(B,D)` | fixed, documented feature representation used by every baseline and T-SOHO (e.g. identity or row L2 normalization). It must never depend on task/time. |
| `Y` | `(B,C)` | global one-hot labels, with original dataset class IDs as columns. |
| `G_t` | `(D,D)` | `Σ_{s≤t} h(X_s)^T h(X_s)`. |
| `Q_t` | `(D,C)` | `Σ_{s≤t} h(X_s)^T Y_s`. |
| `n_t` | `(C,)` | per-class sample count; zero for unseen classes. |
| `μ_c` | `(D,)` | `Q_t[:,c]/n_t[c]`, for `c∈S_t`. |
| `A_t,L_t` | `(C_t,C_t)` | confusion affinity and graph Laplacian over a canonical sorted order of `S_t`. |
| `E_t` | `(r,C)` | global-index class code; columns for `S_t` contain graph code, unseen columns are zero/not scored. |
| `P_t` | `(D,r)` | transportable code regressor. |
| `z` | `(B,r)` | projected feature `h(x)P_t`. |
| `ℓ` | `(B,C)` | global logits; unseen-class entries are masked from argmax. |

`r` must satisfy `1 ≤ r < C_t−1`. T-SOHO has no meaningful graph code before at least three observed classes. The implementation must declare a deterministic bootstrap policy (for example, defer scoring until this condition or use a separately documented two-class fallback); it may not silently treat a full-rank code as the proposed method.

## Streaming update and learner state

For each arriving batch `(X,Y)`, compute `H=h(X)` once, then update only:

```text
G ← G + HᵀH                    # (D,D)
Q ← Q + HᵀY                    # (D,C)
n ← n + sum_rows(Y)            # (C,)
seen_mask ← seen_mask OR (n > 0)
```

The allowed persistent learner state is bounded by `O(D²+DC+C²)` and consists of `G`, `Q`, `n`, `seen_mask`, fixed configuration, selected ridge coefficient/policy, and optionally recomputable current `E`, `P`, and a class-score mask. It may include scalar metadata such as a version, seed, and class-order hash.

The learner state and any checkpoint **must not** contain raw images, dataset indices, paths, batches, a replay buffer, `memory_features`, per-example labels, or a tensor/list whose leading dimension grows with the number of seen examples. Caching such data outside the learner state still violates the claimed exemplar-free protocol if it is needed to resume/train the method.

## Confusion graph and spectral code

For observed classes in canonical increasing global-ID order, choose a documented distance representation. The default candidate is normalized class means:

```text
u_c = μ_c / max(||μ_c||₂, ε)
d²_ij = ||u_i - u_j||₂²
A_ij = exp(-d²_ij / τ) for i≠j; A_ii = 0
D_A = diag(A 1)
L = D_A - A
```

`τ>0` is selected without using test labels. `A` and `L` must be symmetric to numerical tolerance. Let `v_1,…,v_Ct` be orthonormal eigenvectors of `L` in nondecreasing eigenvalue order. The constant eigenvector is excluded. The code is

```text
E_seen = [v_2, …, v_(r+1)]ᵀ       # (r,C_t)
E[:, S_t] = E_seen; E[:, outside S_t] = 0
```

so `E Eᵀ = I_r` (within tolerance). Sign changes and rotations within repeated-eigenvalue eigenspaces must not change decoded logits when `P` is recomputed from the same `E`; implementation should nevertheless use a documented canonical ordering/sign convention for reproducible serialization. Disconnected graphs and eigenvalue multiplicities require an explicit policy and test coverage.

## Analytic training and inference

With an isotropic ridge coefficient `λ>0`, recompute after each task (or a documented update boundary):

```text
P_t = (G_t + λI_D)^−1 Q_t E_tᵀ                 # (D,r)
z = h(x) P_t                                   # (B,r)
ℓ_c = 2 z e_c − ||e_c||₂²  for c∈S_t           # (B,C_t)
ŷ = argmax_{c∈S_t} ℓ_c
```

This is a global classifier. `task_id` may select a stream during training or a reporting subset during evaluation, but it must never select a head, code, classifier, or inference mask beyond the global seen-class mask.

`λ` is either a pre-registered value or selected by a train-only policy shared with controls. If GCV is used, state precisely which sufficient-statistic-compatible approximation is used; it must not require saved examples or use the test set.

## Invariants and required checks

1. **Sufficiency:** sequential batch updates of `G,Q,n` equal their concatenated-data reference within floating-point tolerance.
2. **State:** retained state contains no sample-level object and has no growth proportional to `N_seen`.
3. **Geometry:** `A=Aᵀ`, `L=Lᵀ`, `E Eᵀ≈I_r`, `r<C_t−1`, and class-column mapping is canonical.
4. **Analytic solution:** `P` solves `(G+λI)P=QEᵀ` to tolerance.
5. **Task-free inference:** predictions are an argmax over all observed global class IDs with no task input.
6. **Determinism:** fixed seed, input statistics, configuration, and numerical device policy reproduce state/logits within a stated tolerance.
7. **Checkpoint audit:** saving/loading preserves only permitted statistics and model/config metadata; a checkpoint scan must find no sample-level replay data.

## Relation to raw Ridge and falsifiable hypotheses

Define the matched raw-feature Ridge control `W_R=(G+λI)^−1Q`, shape `(D,C)`. Then

```text
P = W_R Eᵀ
2zE = 2 h(x) W_R (EᵀE)
```

Therefore `W_R EᵀE` is the T-SOHO linear score component (up to the decoder's factor 2), not the entire decoder: the logits also include the per-class term `−||e_c||²`.

The following claims are deliberately falsifiable:

| Hypothesis | Falsifier / mandatory control |
|---|---|
| Strict low-rank graph projection improves the accuracy–state–time trade-off against matched raw Ridge. | No improvement relative to raw Ridge under identical features/splits/λ policy, or improvement disappears when `r` varies. |
| Confusion geometry matters, rather than arbitrary rank reduction. | Random orthonormal code, PCA/label-independent code, and shuffled graph controls match the result. |
| Benefit is not an orthogonal reparameterization. | Full orthogonal `E` has `EᵀE=I`; it matches raw Ridge apart from specified decoder bias. |
| Benefit is not full-simplex ETF argmax equivalence. | For full simplex `E`, `EᵀE=I−11ᵀ/C`; with equal column norms it only subtracts a common score and has raw-Ridge-equivalent argmax. |
| T-SOHO is exemplar-free. | Any checkpoint/state artifact contains images, indices, historical embeddings, labels, or a per-example replay cache. |
| Method is deterministic. | Same statistics/configuration produce materially different code/projector/logits beyond declared numerical tolerance. |

Dynamic Top-K is intentionally outside T-SOHO's transport claim: because WTA selection depends on each sample, it cannot generally be transported exactly by one common linear matrix.
