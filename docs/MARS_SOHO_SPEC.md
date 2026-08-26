# MARS-SOHO specification

Status: Phase-1 research prototype. This specification does not authorize a
held-out test run or SRQ integration.

## Research claim under test

MARS-SOHO replaces the historical frozen-feature replay required by dynamic
SOHO with deterministic distributional replay from class aggregates. It does
not claim that first and second moments determine hard Top-K statistics.

The proposed contribution is the combination of:

1. exact streaming spherical class moments;
2. heterogeneous class reconstruction using class diagonal variance and a
   shrinkage pooled correlation;
3. paired deterministic pseudo-directions regenerated from `(seed,class,j)`;
4. a hard-WTA support certificate that allocates a fixed replay budget;
5. a future, separately gated SRQ square-root state after replay reconstruction
   is validated.

Gaussian pseudo-replay from class means and a shared covariance is an existing
idea and is included only as a control. In particular, AnaCP (NeurIPS 2025)
already combines analytic adaptive projection with Gaussian pseudo-replay.

## Phase-1 state

For normalized frozen features `u=x/||x||`, the learner persists:

```text
class_ids             (C_seen, metadata)
n                      (C_seen,)
s = sum u              (D,C_seen)
d = sum u*u            (D,C_seen)
S_w                    (D,D)
global_sum             (D,)
random projection W    (M,d_olda)
active rotation R      (d_olda,D)
G                      (M,M)       # exact float Phase-1 state
Q, classifier          (M,C_seen)
```

There is no image, historical feature, per-sample label, WTA-code history,
pseudo-feature or saved noise vector. Phase 1 intentionally keeps dense exact
`G`; SRQ compression is a Phase-2 variable.

## Reconstruction

For class `c`, define

```text
mu_c = s_c / n_c
v_c  = max(d_c/n_c - mu_c*mu_c, 0)
```

Let `C_pool` be the correlation corresponding to pooled `S_w`, shrunk toward
identity. Its top `q` eigenpairs plus a diagonal residual define a bounded-rank
factor. Pseudo-directions are

```text
u_tilde(c,j) = normalize(mu_c + diag(sqrt(v_c)) r(c,j))
r(c,j)       = low_rank_correlated_noise + diagonal_residual_noise
```

The control `shared_gaussian` replaces `v_c` by the shared pooled diagonal.
Antithetic Gaussian pairs use a deterministic local generator. Generated
directions are transient and are deleted after reconstructing statistics.

Old-class statistics at a refresh are

```text
z_cj  = TopK(W R_t u_tilde(c,j))
G_old = sum_c (n_c/J_c) sum_j z_cj z_cj^T
Q_old[:,c] = (n_c/J_c) sum_j z_cj
```

The arriving task is encoded exactly once under `R_t`; its exact `G,Q` terms
are added to old-class reconstructed terms and the batch is discarded.

## Support certificate and allocation

For old/new expanded activations `a,b`, let

```text
gamma_k(a) = a_(k) - a_(k+1)
delta      = ||b-a||_inf
```

If `gamma_k(a)>2 delta`, the Top-K support is unchanged. A pilot set estimates
per-class uncertified mass `p_c`. Under a fixed total pseudo budget, Phase 1
uses

```text
J_c proportional to n_c sqrt(p_c + epsilon).
```

`shuffled_support` permutes `p_c` deterministically while preserving the same
budget and distribution model. Thus it tests whether semantic association
between class and boundary risk matters.

## Gauge policy

SOHO eigenspaces with repeated eigenvalues have non-identifiable bases. MARS
aligns only such blocks to the preceding basis using orthogonal Procrustes:

- ETF fixes the discriminative block using class geometry; only its null block
  is aligned;
- without ETF, nearly equal discriminative eigenvalue blocks and the null block
  are aligned independently;
- unequal-eigenvalue subspaces are never mixed.

This prevents support-risk from conflating arbitrary numerical basis rotation
with genuine representation change.

## Oracle and controls

`exact_replay_oracle` retains historical frozen features and re-encodes all of
them. It shares the exact MARS map, fixed ridge coefficient and numeric policy,
so its gap to MARS isolates replay approximation. It is not exemplar-free and
is not identical to the repository's current SOHO fidelity adapter.

The Phase-1 control set is:

| Method | Purpose |
|---|---|
| exact replay oracle | Approximation upper reference under the same new map |
| shared Gaussian | Closest simple moment-replay baseline |
| heterogeneous spherical | Tests class-specific diagonal variance |
| support aware | Proposed fixed-budget allocation |
| shuffled support | Falsifies class-risk association |

Official SOHO replay, official/tuned FLY and raw Ridge remain required in the
eventual final comparison, but do not choose Phase-1 reconstruction settings.

## Train-only selection

Each class is split once into nested partitions:

```text
full train
  outer validation: 20%, untouched until candidate locking
  development: 80%
    inner validation: 20% of development
    inner fit:         80% of development
```

All methods use identical class orders, task splits and development replicates.
Ridge lambda is selected once with the exact replay oracle, then shared by all
reconstruction controls. Shared and heterogeneous models receive the same
rank/shrinkage grid. Support-aware and shuffled controls inherit the selected
heterogeneous configuration; their only difference is allocation.

No `test.pt` may be visible to the runner.

## Falsifiable gates

Per-dataset evidence records:

1. support-aware outer-validation AIA gap to exact oracle at most 0.5 pp;
2. support-aware gain over shared Gaussian at least 0.2 pp;
3. support-aware gain over shuffled support at least 0.1 pp;
4. held-out test cache remained absent.

Phase 2 is not authorized merely because one dataset passes. The cross-dataset
review must show the oracle-gap and shared-Gaussian gates on at least two of
three datasets, no unexplained catastrophic regression, and a valid
exemplar-free state audit.

## Limitations

- Moment replay is model-based and cannot reproduce arbitrary Top-K statistics.
- The support theorem certifies unchanged support for generated directions, not
  unseen real features from a misspecified class distribution.
- Phase-1 expansion width is 1,000 for feasibility; conclusions do not yet
  establish behavior at the FLY/SOHO width 10,000.
- Persistent bytes in Phase 1 are not the expected final memory result because
  dense `G` has not yet been replaced by SRQ.
- A pass does not prove superiority to FLY and does not authorize test tuning.
