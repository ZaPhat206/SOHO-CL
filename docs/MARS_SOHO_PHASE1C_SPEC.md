# MARS-SOHO Phase 1C reconstruction-fidelity specification

Status: train-only diagnostic and candidate gate. It does not authorize a
continual learner, held-out test evaluation, other datasets or SRQ integration.

## Problem isolated by Phase 1B

Changing pseudo-budget allocation had negligible effect while moment replay
remained about 1.8 AIA points below exact feature replay. The current ambient
Gaussian is fitted to unit-normalized features and then normalized again after
sampling; that nonlinear projection need not preserve the stored mean or
covariance. Its pooled correlation also cannot represent class-specific
covariance orientation.

Phase 1C measures reconstruction bias directly before another continual run.

## Proposed representation

For class `c`, let

```text
m_c = normalize(mean(x | c)),  ||x|| = 1.
```

Each feature is mapped to the tangent plane of the sphere:

```text
v = Log_{m_c}(x).
```

The aggregate sketch stores:

```text
n_c                         scalar
m_c                         (D,)
||mean(x|c)||               scalar
mean(v|c)                   (D,)
U_c, lambda_c               (D,r), (r,)
diagonal covariance residue (D,)
calibration scale           scalar
```

`U_c,lambda_c` are obtained by deterministic randomized subspace iteration and
truncation. The exact covariance diagonal minus the represented low-rank
diagonal is retained as a nonnegative residual. State size is `O(C D r)`, has
no sample-count dimension and stores no feature row.

Pseudo tangent vectors are generated deterministically and mapped back with

```text
x_tilde = Exp_{m_c}(scale_c * v_tilde).
```

For the calibrated candidate, `scale_c` is found by deterministic bisection so
that a disjoint 256-direction calibration stream matches the stored resultant
length `||mean(x|c)||`. This calibration does not use validation or test data.

## Fidelity objective

On validation features under the same locked SOHO map, define count-scaled
targets `G*` and `Q*`. For reconstructed statistics `G_hat,Q_hat`, report

```text
e_G = ||G_hat-G*||_F / ||G*||_F
e_Q = ||Q_hat-Q*||_F / ||Q*||_F
e   = (e_G+e_Q)/2.
```

The empirical fit-feature replay oracle reports the finite-sample reference.
All methods also solve the same Ridge system and evaluate validation accuracy.
Feature mean, diagonal second-moment and resultant-length errors are reported
separately.

## Nested train-only protocol

For each of three new split/projection replicate pairs, every class is split:

```text
full train
  outer validation: 20%
  development:      80%
    inner validation: 20% of development
    inner fit:         80% of development
```

Only tangent rank `{8,16,32}` is selected, using mean inner combined statistic
error. A near tie within `0.002` chooses the lower rank. Accuracy does not
select the candidate. The selected rank is then evaluated once on untouched
outer validation.

All methods use width 1,000, 64 pseudo-directions per class, Ridge lambda 10,
the same backbone, SOHO map, partitions and projection seed. Controls are:

- empirical fit-feature replay oracle (not exemplar-free);
- Phase-1 ambient spherical reconstruction;
- tangent low-rank without resultant calibration;
- tangent low-rank with resultant calibration (proposal).

## Gates

The calibrated tangent candidate must satisfy all of:

1. at least 10% relative reduction in combined statistic error versus ambient;
2. at least 0.20 pp validation-accuracy gain over ambient;
3. at most 0.50 pp accuracy gap to empirical replay oracle;
4. mean resultant-length error at most 0.02;
5. zero persistent sample-level bytes;
6. `test.pt` remains absent.

Gates are not relaxed after the run. A pass authorizes a separate review for
streaming learner integration; it does not itself constitute continual-learning
or held-out evidence.

## Limitations

- Phase 1C fits each class sketch once; streaming merge-and-truncate is not yet
  implemented or claimed.
- The randomized sketch is an approximation, and the exponential-map Gaussian
  does not match arbitrary spherical distributions.
- Resultant calibration matches one scalar moment, not the full distribution.
- The empirical replay oracle and validation target differ by finite-sample
  variation, so zero statistic error is neither expected nor required.
