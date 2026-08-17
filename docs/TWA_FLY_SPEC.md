# TWA-FLY specification

## Scope and claim boundary

Two-Way Analytic FLY (TWA-FLY) is an opt-in, exemplar-free extension of the
fixed-backbone FLY pipeline. It couples the raw frozen-backbone view to the
fixed FlyHash/WTA view during classifier fitting. Inference still uses only the
FLY view and never accepts a task identifier.

The method is inspired by the two-way consistency principle in BiCyc, but it is
not an implementation of BiCyc. BiCyc compensates representation drift between
old and new trainable feature spaces. TWA-FLY has a frozen backbone and fixed
FlyHash map, so no such temporal feature spaces exist. Its mathematical family
is co-regularized multi-view least squares. Novelty must therefore be evaluated
as the combination of exact streaming sufficient statistics, nonlinear WTA/raw
cross-view coupling, baseline-preserving inference, and the continual-learning
memory contract - not as a generic claim that agreement regularization is new.

Primary references used to establish this boundary include:

- Fly-CL, ICLR 2026: <https://openreview.net/pdf/740f2f76225c5b2228b9a6f6da54e0fc26a32dfa.pdf>
- BiCyc, ICLR 2026: <https://openreview.net/pdf?id=7UfZAxKo5K>
- F-OAL, NeurIPS 2024: <https://openreview.net/pdf?id=rGEDFS3emy>
- Multi-view co-regularization framework: <https://arxiv.org/abs/1401.8066>
- Statistical regularization theory for continual learning, ICML 2024:
  <https://proceedings.mlr.press/v235/zhao24n.html>

## Fixed representation

At task `t`, let:

- `X_t in R^(N_t x D)` be frozen-backbone features;
- `S in R^(H x D)` be the fixed sparse Gaussian FlyHash projection;
- `Z_t = TopK(X_t S^T) in R^(N_t x H)` use the existing FLY signed-value,
  largest-value Top-K semantics;
- `Y_t in R^(N_t x C)` be global one-hot targets with a fixed output width;
- `D=768`, `H=10000`, `C=100` in the locked CIFAR-100 pilot.

TWA-FLY never attempts to linearly transport Top-K. It evaluates Top-K once for
each arriving sample and immediately discards the sample after updating the
statistics.

## Streaming sufficient statistics

The learner retains only:

```
G_xx = sum_t X_t^T X_t       shape (D, D)
G_zz = sum_t Z_t^T Z_t       shape (H, H)
R_xz = sum_t X_t^T Z_t       shape (D, H)
Q_x  = sum_t X_t^T Y_t       shape (D, C)
Q_z  = sum_t Z_t^T Y_t       shape (H, C)
n     = sum_t bincount(y_t)   shape (C,)
```

No row-indexed tensor of shape `N_seen x ...` is persistent learner state.
Feature and WTA caches used to accelerate controlled experiments are explicitly
experiment infrastructure and must never be serialized in a learner checkpoint.

## Symmetric objective

For raw-view weights `U in R^(D x C)` and deployed FLY weights
`V in R^(H x C)`, solve

```
min_U,V ||XU-Y||_F^2 + ||ZV-Y||_F^2
        + rho ||XU-ZV||_F^2
        + lambda_x ||U||_F^2 + lambda_z ||V||_F^2 .
```

The block normal equations are

```
[(1+rho)G_xx + lambda_x I_D,  -rho R_xz    ] [U] = [Q_x]
[-rho R_xz^T,                 (1+rho)G_zz + lambda_z I_H] [V]   [Q_z].
```

The implementation must use linear solves, never an explicit inverse. It uses
deterministic alternating exact block minimization:

```
U <- solve(A_x, Q_x + rho R_xz V)
V <- solve(A_z, Q_z + rho R_xz^T U)
```

where `A_x=(1+rho)G_xx+lambda_x I` and
`A_z=(1+rho)G_zz+lambda_z I`. Both Cholesky factors are reused. Iteration stops
when the relative block-equation residual is at most the configured tolerance,
or fails closed at the iteration limit.

## One-way control

The raw teacher is first fit independently:

```
U_0 = solve(G_xx + lambda_x I, Q_x).
```

Then only the deployed FLY classifier is updated:

```
V = solve((1+rho)G_zz + lambda_z I, Q_z + rho R_xz^T U_0).
```

This distinguishes symmetric mutual correction from ordinary teacher-to-student
logit matching.

## Inference

```
z = TopK(x S^T)
logits = z V                     shape (batch_size, C)
prediction = argmax(logits)
```

The raw branch is not evaluated at inference. There is no task ID, task router,
or stored historical sample.

## Invariants and mathematical checks

1. Streaming statistics equal their batch definitions for any partition and
   arrival order, up to floating-point accumulation error.
2. For positive `lambda_x` and `lambda_z` and `rho >= 0`, the objective is a
   strictly convex quadratic; its block system is symmetric positive definite
   and has a unique solution.
3. Exact alternating block minimization is deterministic, monotonically
   non-increasing in objective value, and converges to that unique solution.
4. At `rho=0`, `V=solve(G_zz+lambda_z I,Q_z)` exactly recovers the matched FLY
   Ridge classifier. This is a mandatory implementation identity test.
5. The alternating result must match a direct small-dimensional block solve in
   synthetic tests.
6. The final relative block residual must be finite and at most `1e-5`.
7. Persistent tensor shapes may depend only on `D`, `H`, and `C`, never on the
   historical sample count.
8. A checkpoint contains aggregate statistics and configuration only. Derived
   weights may be recomputed deterministically after loading.

## Hyperparameter protocol

- FLY projection, Top-K, class order, preprocessing, backbone, seed, and current
  task GCV policy are inherited exactly from the matched FLY implementation.
- `lambda_z` is the same current-task GCV choice used by matched FLY at each
  stage.
- `lambda_x` is locked before the study; it is not selected on held-out test.
- Only `rho` is selected in the Phase A train-only pilot.
- Test features are physically hidden while selecting `rho`.

## Falsifiable hypotheses and Phase A gate

The primary hypothesis is that raw/WTA agreement corrects WTA decisions lost by
the sample-dependent Top-K bottleneck without paying raw-view inference cost.
It is falsified on the locked train-only pilot unless all conditions hold:

- symmetric TWA-FLY exceeds matched FLY validation average incremental
  accuracy by at least `0.20` percentage point;
- symmetric TWA-FLY exceeds the one-way control by at least `0.10` point;
- symmetric TWA-FLY exceeds a destroyed-correspondence cross-statistic control
  by at least `0.10` point;
- final relative block residual is at most `1e-5` for every task;
- persistent learner state is at most `1.10` times matched FLY state;
- the held-out `test.pt` remains physically unavailable throughout selection.

Passing this gate does not establish a paper claim. It only authorizes a
separately locked held-out evaluation. Failure means stop; do not tune against
test or silently change the gate.

## Limitations

- Agreement may merely shrink both predictors toward a compromise and can hurt
  when raw and WTA errors are correlated.
- The cross statistic costs `O(DH)` memory; TWA-FLY reduces neither the dominant
  `O(H^2)` FLY Gram matrix nor FLY update complexity.
- There is no theorem that agreement improves classification accuracy.
- The construction relies on paired raw/WTA views at arrival time. It cannot be
  reconstructed later from marginal `G_xx` and `G_zz` alone.
- The result does not make dynamic Top-K linearly transportable and does not
  inherit BiCyc's feature-drift guarantees.

## Post-Phase-A status

The locked Phase A train-only gate failed. Agreement-only TWA-FLY is retained
as a negative result/control and must not proceed to held-out evaluation. The
separate D0 protocol in `docs/research/TWA_FLY_D0_PROTOCOL.md` measures raw/FLY
error complementarity before any joint residual method is considered. D0 is not
a continuation or post-hoc retuning of the Phase A rho search.
