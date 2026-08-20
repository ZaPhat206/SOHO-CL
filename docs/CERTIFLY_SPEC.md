# CertiFLY specification

Status: implementation contract for the Q0 mathematical gate. CertiFLY is a
new method and must not modify the existing FLY or SOHO implementations.

## Objective

CertiFLY keeps the fixed FLY representation and global analytic classifier:

\[
z=\operatorname{TopK}(Wx),\qquad
G_t=G_{t-1}+Z_t^\top Z_t,\qquad
Q_t=Q_{t-1}+Z_t^\top Y_t.
\]

It compresses the quadratic persistent state `G:(m,m)` without reducing its
coordinate dimension or intentionally truncating its spectrum. The classifier
is rebuilt with a linear solve, never an explicit inverse:

\[
\widehat C_t=(\widehat G_t+\lambda_t I)^{-1}Q_t.
\]

## Symmetric correlation quantization

Let `g=diag(G)` and `d=sqrt(g)`. For active coordinates define

\[
R=D^\dagger G D^\dagger,\qquad D=\operatorname{diag}(d),
\]

where a zero diagonal uses a zero reciprocal. CertiFLY stores `g` exactly in
the configured statistics dtype and stores only the strict upper triangle of
`R`. Upper-triangular blocks are symmetrically quantized:

\[
s_B=\frac{\max_{u\in B}|u|}{2^{b_B-1}-1},\qquad
q_B=\operatorname{round}(B/s_B),\qquad b_B\in\{8,16\}.
\]

The diagonal of `R` is not quantized. Reconstruction explicitly mirrors the
upper triangle, so `G_hat` is exactly symmetric in storage semantics.

All blocks start at int8. Blocks with the largest reduction in weighted
Frobenius error are promoted deterministically to int16 until the configured
certificate budget is met. If the budget cannot be met, the update fails; it
must not silently fall back to a different model.

## Streaming merge and certificate

At task `t`, the previous compressed state is reconstructed, the exact current
task increment `Delta_G_t=Z_t^T Z_t` is added, and the result is requantized.
No row of an earlier `Z` is retained. If `epsilon_(t-1)` bounds the discrepancy
between the ideal cumulative Gram and the reconstructed previous state, and
`eta_t` bounds the current requantization error, then

\[
\epsilon_t=\epsilon_{t-1}+\eta_t
\]

is a valid deterministic bound by the triangle inequality. CertiFLY uses
`eta_t=||E_t||_F`, hence also bounds the spectral error because
`||E_t||_2 <= ||E_t||_F`.

For `A=G+lambda I`, `G_hat=G+E`, if `epsilon_t < lambda`, then `G_hat+lambda I`
is positive definite and

\[
\frac{\|\widehat C-C\|_2}{\|C\|_2}
\leq \frac{\epsilon_t}{\lambda-\epsilon_t}.
\]

For a query code `z`, a uniform per-logit perturbation bound is

\[
\delta(z)=\|z\|_2
\frac{\epsilon_t}{\lambda(\lambda-\epsilon_t)}\|Q\|_2.
\]

An exact-FLY top-1 prediction with margin greater than `2*delta(z)` is certified
unchanged by the quantized classifier. This is a sufficient, potentially
conservative condition; failure to certify is not proof that the prediction
changed.

## Learner state

Allowed persistent tensors are:

- deterministic sparse FLY projection (or its seed/configuration);
- exact Gram diagonal;
- quantized strict-upper blocks and one scale per block;
- `Q:(m,C_seen)`, class counts and sorted class mapping;
- derived classifier `C_hat:(m,C_seen)`;
- scalar/block metadata and cumulative error bound.

Forbidden state includes images, raw backbone features, historical WTA codes,
labels aligned per sample, sample indices, or any tensor whose dimension is the
historical sample count. Feature/WTA caches used by an experiment runner are
disk infrastructure and must never be serialized in a learner checkpoint.

Inference accepts only samples/features/codes, never a task ID.

## Q0 invariants

1. Quantization and reconstruction are deterministic and symmetric.
2. The exact diagonal is preserved.
3. Every measured Gram error is below the reported cumulative bound.
4. A certified system is positive definite and is solved without inversion.
5. The classifier perturbation and argmax certificates hold on synthetic data.
6. Save/resume reproduces logits and future updates.
7. Persistent state contains no sample-level tensor.
8. With `m=10000`, `C=200`, block size 256 and all-int8 Gram blocks, the
   projected persistent state is at most 25% of matched exact FLY state under
   the same state-accounting policy.

## Falsifiable hypotheses

- H1: int8/int16 adaptive storage remains certified under the locked FLY Ridge
  range on real streams.
- H2: validation average accuracy is within 0.50 percentage point of exact FLY.
- H3: persistent learner state is at most 25% of exact FLY.
- H4: mixed precision materially outperforms rank-truncated controls at matched
  state because it retains every WTA coordinate and off-diagonal location.
- H5: quantization/update overhead does not erase the deployment benefit. The
  first prototype claims persistent-state compression only; peak runtime memory
  and latency are reported separately and are not assumed to improve.

