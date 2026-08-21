# SRQ-FLY proof appendix

Status: algebraic appendix for review. These statements concern the implemented
fixed-code analytic learner. They do not prove an accuracy improvement or a
privacy guarantee.

## A. Notation and implemented recursion

Let \(Z_t\in\mathbb R^{n_t\times m}\) contain the fixed FLY/WTA codes arriving
at stage \(t\), and let \(Y_t\in\mathbb R^{n_t\times C_t}\) be their global
one-hot targets after deterministic class-column expansion. Define

\[
G_t=\sum_{k=1}^t Z_k^\top Z_k,
\quad Q_t=\sum_{k=1}^t Z_k^\top Y_k,
\quad A_t=G_t+\lambda I_m,
\]

with \(\lambda>0\). Missing target columns in earlier tasks are zeros.

Let \(\mathcal C\) encode an upper-triangular factor with an unquantized
float32 diagonal and a deterministic groupwise-int8 strict triangle, and let
\(\mathcal D\) decode it. The implemented recursion is

\[
B_t=\begin{cases}
\lambda I_m+Z_1^\top Z_1,&t=1,\\
\widehat R_{t-1}^\top\widehat R_{t-1}+Z_t^\top Z_t,&t>1,
\end{cases}
\]

\[
R_t=\operatorname{chol}(B_t)^\top,
\qquad \widehat R_t=\mathcal D(\mathcal C(R_t)).
\]

The classifier actually used by SRQ is

\[
\widehat W_t=(\widehat R_t^\top\widehat R_t)^{-1}Q_t,
\]

implemented by two triangular solves. Compression occurs after every stage,
so \(\widehat R_t^\top\widehat R_t\) generally differs from \(A_t\).

## B. Exact streaming statistics

**Lemma 1.** Repeated updates

\[
G_t=G_{t-1}+Z_t^\top Z_t,
\qquad Q_t=\operatorname{expand}(Q_{t-1})+Z_t^\top Y_t
\]

equal batch accumulation over all codes and targets observed through stage
\(t\).

**Proof.** The claim is immediate for \(t=1\). Assume it holds at \(t-1\).
Class-column expansion appends zero columns and leaves existing columns
unchanged, so it embeds the historical sum in the current global label space.
Adding the current products gives the sums through stage \(t\). Induction
completes the proof. \(\square\)

This lemma requires the codes themselves to remain fixed. It does not recover
historical dynamic-WTA codes from raw-feature means or covariances.

## C. Exact square-root equivalence without compression

**Theorem 1.** Suppose \(\lambda>0\), arithmetic is exact, and
\(\mathcal D\circ\mathcal C\) is the identity. Then

\[
R_t^\top R_t=\lambda I_m+\sum_{k=1}^t Z_k^\top Z_k=A_t
\]

for every stage \(t\).

**Proof.** At \(t=1\), the definition of the Cholesky factor gives the claim.
If it holds at \(t-1\), then

\[
B_t=R_{t-1}^\top R_{t-1}+Z_t^\top Z_t
=\lambda I_m+\sum_{k=1}^t Z_k^\top Z_k.
\]

The right-hand side is positive definite because \(\lambda>0\). Its Cholesky
factor exists and satisfies \(R_t^\top R_t=B_t\). \(\square\)

This is an uncompressed control. Finite precision and encode/decode make SRQ
an approximate recursion.

## D. Positive definiteness by construction

**Theorem 2.** If an upper-triangular matrix \(\widehat R\) has finite,
strictly positive diagonal entries, then
\(\widehat A=\widehat R^\top\widehat R\) is symmetric positive definite.

**Proof.** A triangular matrix with nonzero diagonal is nonsingular. For any
nonzero \(x\), \(\widehat R x\ne0\), hence

\[
x^\top\widehat A x
=x^\top\widehat R^\top\widehat R x
=\|\widehat R x\|_2^2>0.
\]

Therefore \(\widehat A\succ0\). \(\square\)

Strict-upper quantization cannot change the diagonal. The decoder validates
finite positive scales, and the learner checks the reconstructed diagonal
before solving. This guarantees structural positive definiteness, not a small
condition number or a small prediction error.

## E. From factor error to Gram error

Let \(R\) be an exact upper factor and \(\widehat R=R+E\). Then

\[
\Delta=\widehat R^\top\widehat R-R^\top R
=R^\top E+E^\top R+E^\top E,
\]

and submultiplicativity gives

\[
\|\Delta\|_2\le(2\|R\|_2+\|E\|_2)\|E\|_2.
\]

This is a local relation. In SRQ, earlier factor errors enter later \(B_t\),
so a stream-level result also needs a recurrence for \(\|E_t\|\). The current
manuscript does not claim non-accumulation.

## F. Ridge classifier perturbation

**Theorem 3.** Let \(A\succ0\), \(\widehat A=A+\Delta\succ0\), and
\(W=A^{-1}Q\), \(\widehat W=\widehat A^{-1}Q\). Then

\[
\|\widehat W-W\|_F
\le
\frac{\|\Delta\|_2\|Q\|_F}
{\lambda_{\min}(A)\lambda_{\min}(\widehat A)}.
\]

If \(\|\Delta\|_2<\lambda_{\min}(A)\), then

\[
\|\widehat W-W\|_F
\le
\frac{\|\Delta\|_2\|Q\|_F}
{\lambda_{\min}(A)
[\lambda_{\min}(A)-\|\Delta\|_2]}.
\]

**Proof.** The resolvent identity is

\[
\widehat A^{-1}-A^{-1}=-A^{-1}\Delta\widehat A^{-1}.
\]

Multiply by \(Q\), apply spectral/Frobenius submultiplicativity, and use
\(\|A^{-1}\|_2=1/\lambda_{\min}(A)\). This proves the first bound. Weyl's
inequality gives
\(\lambda_{\min}(\widehat A)\ge
\lambda_{\min}(A)-\|\Delta\|_2\), proving the second. \(\square\)

For a code vector \(z\),

\[
\|z^\top(\widehat W-W)\|_2
\le\|z\|_2\|\widehat W-W\|_F.
\]

## G. Conditional argmax preservation

Let \(c^*=\arg\max_c z^\top w_c\) be the unique exact winner and define

\[
\gamma(z)=z^\top w_{c^*}-\max_{c\ne c^*}z^\top w_c>0.
\]

If every class logit changes by at most \(\varepsilon(z)\) and
\(2\varepsilon(z)<\gamma(z)\), then SRQ preserves the winner.

**Proof.** The winner can decrease by at most \(\varepsilon\), while a
competitor can increase by at most \(\varepsilon\). The perturbed gap is at
least \(\gamma-2\varepsilon>0\). \(\square\)

This condition is sample-specific. It does not retroactively pass D4's 98%
agreement gate.

## H. Exact persistent tensor-byte accounting

For the repository's CSC projection with \(\nu\) actually stored float32
nonzeros (nominally \(ms\)) and feature dimension \(d\),

\[
B_H=4\nu+8\nu+8(d+1),
\]

for values, row indices, and column pointers. For int8 SRQ with block set
\(\mathcal B\), strict-upper entry count \(N=m(m-1)/2\), block payload sizes
\(n_b\), and group size \(g\),

\[
B_R=4m+N+4\sum_{b\in\mathcal B}\left\lceil\frac{n_b}{g}\right\rceil.
\]

Under the locked float32 experiments with \(C\) seen classes,

\[
B_{Q,W,n}=4mC+4mC+4C,
\qquad B_{\mathrm{SRQ}}=B_H+B_R+B_{Q,W,n}.
\]

The state-matched exact FLY total replaces \(B_R\) by \(4m^2\). Quantization
groups restart per block, so one global ceiling is not exact. Python class
mappings and bounded scalar metadata are not included in
`persistent_state_bytes()`, while the retained classifier is included.
Serialized checkpoint size, runtime workspace, and sample-level caches are
separate quantities.

## I. Software-state invariant

Allowed persistent tensors depend only on \(d,m,C,s\), block/group sizes, and
sparse indices. No allowed tensor has a historical-sample axis, and
`predict_logits(features)` has no task-ID parameter. This establishes the
repository's exemplar-free state contract. It does not imply differential
privacy, resistance to statistic inversion, or absence of sample-level data
in external experiment caches.
