# MARS-SOHO Phase 1B specification

Status: preregistered train-only allocation repair. This phase does not
authorize held-out test evaluation or SRQ integration.

## Motivation

Phase 1 showed that the sufficient hard-WTA support certificate saturated:
every class had certificate-failure risk one, so the proposed fixed-budget
allocator reduced to uniform replay. Phase 1B preserves that certificate as a
diagnostic and replaces its allocation role with a quantity tied directly to
the Monte-Carlo estimator of SOHO's analytic statistics.

## Statistics and theorem

For class `c`, current-map WTA code `z`, and conceptual sufficient-statistic
vector

```text
s(z) = (vec(zz^T), z),
```

the class contribution is `T_c = n_c E[s(z)]`. With `J_c` independent pseudo
draws, the unbiased estimator has squared-error expectation

```text
E ||T_hat - T||^2 = sum_c n_c^2 V_c / J_c,
V_c = E ||s(z) - E s(z)||^2.
```

Under `sum_c J_c = B` and positive real allocations, the Lagrange optimum is

```text
J_c = B n_c sqrt(V_c) / sum_j n_j sqrt(V_j).
```

The implementation enforces an integer minimum per class and allocates the
remaining budget by largest remainders. It estimates a dimensionless equal
weight combination of Gram and cross relative variance:

```text
V_G = (E||z||^4 - ||E[zz^T]||_F^2) / E||z||^4
V_Q = (E||z||^2 - ||E[z]||^2)       / E||z||^2
r_c = (V_G + V_Q) / 2.
```

`||E[zz^T]||_F^2` is computed from the pilot kernel `ZZ^T`, so no per-pilot
`M x M` matrices are materialized. Allocation uses `n_c sqrt(r_c + epsilon)`.

The theorem applies to an ideal independent Monte-Carlo estimator and its
declared squared-statistic objective. The implementation uses deterministic
antithetic pseudo-directions, an estimated risk, integer allocation and a
nonlinear downstream Ridge classifier. Therefore it does not guarantee an
accuracy improvement.

## Disjoint pilot and estimator streams

Allocation pilots and pseudo-statistic samples use disjoint deterministic RNG
streams derived from `(seed, class_id, stream_offset)`. This avoids sizing an
estimator using the same pseudo-directions whose error is being estimated.
Neither stream is persisted in learner state.

## Controls

All controls use the Phase-1 inner-selected `lambda=10`, covariance rank `64`,
shrinkage `0.1`, 64 pseudo-directions per old class in total, and the same
fresh train-only partitions.

| Method | Purpose |
|---|---|
| exact replay oracle | Matched non-exemplar-free upper reference |
| heterogeneous spherical | Uniform allocation control |
| turnover aware | Continuous observed Top-K support replacement |
| shuffled turnover | Association control for turnover |
| statistic variance aware | Preregistered Phase-1B proposal |
| shuffled statistic variance | Association control for estimator variance |

Top-K turnover for one pilot is

```text
1 - |S_old intersect S_new| / K.
```

It is an exact observation for the pilot, not a certificate for unseen real
features. It is retained to diagnose the Phase-1 saturation mechanism; the
statistic-variance method is the primary hypothesis.

## Fresh validation and locked values

Phase 1B uses split seed `3031` and three new class-order/projection replicate
pairs. It does not reuse the consumed Phase-1 outer partitions for its
decision. No hyperparameter is selected in Phase 1B. The values carried from
Phase 1 are checked against the immutable artifact identity
`d1241647...af72f0` before execution.

The runner refuses any feature-cache directory containing `test.pt`.

## Gates

The primary statistic-variance method must satisfy all of:

1. mean AIA gap to exact replay at most `0.50` percentage point;
2. mean AIA gain over uniform heterogeneous replay at least `0.20` point;
3. mean AIA gain over shuffled statistic variance at least `0.10` point;
4. risk spread exceeds `1e-6` on at least 80% of eligible stages;
5. allocations are nonuniform on at least 80% of eligible stages;
6. shuffling changes allocation on at least 80% of eligible stages;
7. `test.pt` remains absent.

A failure is recorded as a negative result. Gates are not relaxed after the
run. Phase 2 SRQ integration remains blocked unless the cross-dataset review
later authorizes it.
