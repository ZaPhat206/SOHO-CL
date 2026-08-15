# PPS-SOHO specification

Status: Phase A implementation contract. PPS-SOHO is selected by its own
configuration and does not change current FLY, SOHO, T-SOHO, SFT, or CRT
semantics.

## Fixed representation

For frozen backbone features `X:(N,D)`, use one fixed FlyHash projection
`R:(H,D)` and fixed WTA rule for the complete stream:

```text
Z = TopK(X R^T):(N,H).
```

The projection, WTA ratio, and seed never change after the first task. Dynamic
pre-WTA transport is forbidden because historical WTA active sets cannot in
general be reconstructed from aggregate state.

## Exact protected statistics

For seen-class indicator `Y:(N,C)`, counts `D_n=diag(n_c)`, and class means

```text
M = Z^T Y D_n^-1:(H,C),
Q = Z^T Y = M D_n:(H,C),
R_w = (I - Y D_n^-1 Y^T) Z:(N,H),
G = Z^T Z = M D_n M^T + R_w^T R_w.
```

`M`, counts, and `Q` are exact. A deterministic Frequent-Directions sketch
`B_w:(ell,H)` approximates only within-class covariance:

```text
R_w^T R_w ~= B_w^T B_w,    ell << H.
```

For a batch of `m` samples from a class with old count `n`, old mean `mu`,
batch mean `mu_b`, feed the batch-centered rows and the merge row

```text
sqrt(n*m/(n+m)) * (mu_b - mu)
```

to the sketch. Their covariance equals the exact Welford scatter increment.

## Classifier

The proposed structured classifier is

```text
W_gamma = solve(M D_n M^T + gamma B_w^T B_w + lambda I, M D_n).
```

`gamma=1` approximates exact FLY Ridge; train-only selection may test other
non-negative values as a falsifiable within-class shrinkage hypothesis. Define

```text
A = concat_rows(sqrt(gamma) B_w, sqrt(D_n) M^T):(ell+C,H).
```

For numerical stability, form an orthonormal basis
`U=orth(concat_columns(A^T,Q))`. The solution lies completely in this compact
space, giving an exact sketched solve without an explicit inverse or an
`H x H` matrix:

```text
W = U solve(U^T A^T A U + lambda I, U^T Q).
```

Sketches and class means may be stored as float32. The compact factorization
and current classifier are promoted to float64 because the matched small-Ridge
regime is numerically unstable in float32; state accounting uses the actual
mixed-precision tensor sizes.

Inference is global and task-free: `logits = TopK(x R^T) W`.

## Persistent learner state

Allowed tensors are:

```text
fixed sparse projection: (H,D)
within-class sketch:     (ell,H)
class means:             (H,C_seen)
counts:                  (C_seen,)
current classifier:      (H,C_seen)
```

No raw image, raw backbone-feature row, WTA-code row, historical label vector,
sample index, replay tensor, or tensor dimension equal to `N_seen` is allowed.
The state complexity is `O(H*ell + H*C_seen)` and is independent of sample
count. This is an exemplar-free claim, not a differential-privacy claim.

## Invariants and theorem targets

1. Streaming class means/counts/cross-products equal their batch values.
2. Welford merge rows reconstruct exact within-class scatter before sketching.
3. Frequent Directions certifies
   `0 <= R_w^T R_w - B_w^T B_w <= Delta I`.
4. The class-mean covariance `M D_n M^T` is never sketched.
5. Compact subspace logits equal a direct solve of the sketched system.
6. If the sketch is exact and `gamma=1`, logits equal exact FLY Ridge.
7. Coefficient error is bounded by
   `||W_hat-W||_F <= gamma*Delta/lambda * ||W||_F`.
8. A prediction is preserved when its exact margin exceeds twice its certified
   maximum logit perturbation.

## Controls and falsifiable gates

The first CIFAR-100 pilot uses cached frozen ViT features and training data
only. It compares raw Ridge, exact FLY at the same `H`, standard global FD, and
class-protected FD. Every method shares projection distribution, `H`, WTA,
class order, split, and seed where applicable.

Proceed to a locked held-out evaluation only if:

1. all synthetic identities and checkpoint/state tests pass;
2. every solver relative residual is at most `1e-4`;
3. protected FD is not more than `0.50` percentage points below exact matched
   FLY on train-only validation;
4. protected FD improves over the best standard-FD control by at least `0.10`
   points, otherwise the class-protection claim is rejected;
5. persistent state is smaller than matched exact FLY;
6. no test cache is opened during selection.

The 1024-dimensional pilot is a correctness/viability gate, not a paper result
and not a replacement for the locked 10,000-dimensional FLY comparison.
