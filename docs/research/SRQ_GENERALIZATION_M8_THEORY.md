# SRQ generalization M8: perturbation theory

Status: theory complete. M8 requires no dataset or Colab execution. Its role
is to connect the local factor quantization measured in M7 to cumulative
system, classifier, logit, and prediction changes without importing an
optimizer-convergence theorem from Shampoo.

## 1. Setup

Let the exact additive Ridge system after task \(t\) be

\[
A_t=A_{t-1}+S_t,\qquad S_t=\Phi_t^\top\Phi_t,\qquad A_0=\lambda I,
\]

and let \(B_t=B_{t-1}+\Phi_t^\top Y_t\). Exact Ridge returns
\(W_t=A_t^{-1}B_t\).

SRQ decodes its previous persistent factor \(\widehat R_{t-1}\), then applies
QR to the stacked matrix

\[
\begin{bmatrix}\widehat R_{t-1}\\\Phi_t\end{bmatrix}.
\]

Denote the unquantized triangular output by \(\bar R_t\). In exact arithmetic,

\[
\bar R_t^\top\bar R_t
=\widehat R_{t-1}^\top\widehat R_{t-1}+S_t.
\]

After mixed-precision storage and decoding, write

\[
\widehat R_t=\bar R_t+E_t,
\qquad \widehat A_t=\widehat R_t^\top\widehat R_t.
\]

Here \(E_t\) is the local factor-storage error at task \(t\). It is distinct
from the cumulative effective-system error
\(\Delta_t=\widehat A_t-A_t\).

## 2. Structural positive definiteness

If the decoded triangular factor \(\widehat R_t\) is nonsingular, then
\(\widehat A_t=\widehat R_t^\top\widehat R_t\) is symmetric positive definite,
because for every nonzero \(v\),

\[
v^\top\widehat A_tv=\|\widehat R_tv\|_2^2>0.
\]

This is a structural guarantee only. It does not imply
\(\widehat A_t=A_t\), a small condition number, or unchanged predictions.

## 3. Exact cumulative-error recurrence

Define the system perturbation introduced by the current encode/decode as

\[
D_t=\bar R_t^\top E_t+E_t^\top\bar R_t+E_t^\top E_t.
\]

Then

\[
\boxed{\Delta_t=\Delta_{t-1}+D_t},\qquad \Delta_0=0.
\]

Proof: expand
\(\widehat A_t=(\bar R_t+E_t)^\top(\bar R_t+E_t)\), use the QR identity above,
and subtract \(A_t=A_{t-1}+S_t\). Therefore,

\[
\Delta_t=\sum_{k=1}^{t}D_k.
\]

For the spectral norm,

\[
\|D_t\|_2
\le 2\|\bar R_t\|_2\|E_t\|_2+\|E_t\|_2^2,
\]

and consequently

\[
\boxed{
\|\Delta_t\|_2
\le\sum_{k=1}^{t}
\left(2\|\bar R_k\|_2\|E_k\|_2+\|E_k\|_2^2\right)}.
\]

If \(\eta_k=\|E_k\|_2/\|\bar R_k\|_2\), each summand is bounded by
\((2\eta_k+\eta_k^2)\|\bar R_k\|_2^2\). This is a worst-case upper bound:
it ignores cancellation between tasks and does not require the realized error
to grow monotonically.

## 4. Ridge-solution perturbation

Assume \(A_t\) is nonsingular and

\[
\epsilon_t=\|A_t^{-1}\Delta_t\|_2<1.
\]

Then \(\widehat A_t=A_t+\Delta_t\) is nonsingular. With
\(\widehat W_t=\widehat A_t^{-1}B_t\), the exact identity

\[
\widehat W_t-W_t
=-(A_t+\Delta_t)^{-1}\Delta_t W_t
\]

and the standard inverse-perturbation bound give

\[
\boxed{
\frac{\|\widehat W_t-W_t\|_F}{\|W_t\|_F}
\le\frac{\epsilon_t}{1-\epsilon_t}}.
\]

Thus a small factor error alone is not sufficient: its accumulated system
effect is filtered through the inverse Ridge system. M7 measures a randomized
action of \(\Delta_t\), not \(\epsilon_t\), so it cannot be used as a numerical
upper bound without an estimate of \(\|A_t^{-1}\|_2\).

## 5. Logit and top-1 prediction consequences

For a row feature \(\phi(x)^\top\), define exact and approximate logits
\(\ell=\phi(x)^\top W_t\) and
\(\widehat\ell=\phi(x)^\top\widehat W_t\). Then

\[
\|\widehat\ell-\ell\|_2
\le \|\phi(x)\|_2\|\widehat W_t-W_t\|_2.
\]

Let \(c^*=\arg\max_c\ell_c\) and let the exact top-1 margin be

\[
\gamma(x)=\ell_{c^*}-\max_{c\ne c^*}\ell_c.
\]

The exact top-1 prediction is preserved whenever

\[
\boxed{2\|\widehat\ell-\ell\|_\infty<\gamma(x)}.
\]

This condition is sufficient, not necessary. A sample that fails the
certificate can still retain the same prediction.

## 6. Connection to M7

- local factor error estimates the relative size of \(E_t\);
- randomized system-action error probes cumulative \(\Delta_t\) on 16 fixed
  directions, but is not \(\|\Delta_t\|_2\);
- weight and logit errors observe downstream perturbation after the solve;
- prediction agreement measures realized decision stability;
- margin-certified fraction evaluates the sufficient condition above.

M7 finds only a 1.016x increase in final local factor error from width 10k to
20k, but 1.79x weight error and 2.00x logit error. This is consistent with
downstream amplification of accumulated system perturbations. It is not proof
of ill-conditioning because M7 does not estimate eigenvalues,
\(\|A_t^{-1}\|_2\), or a condition number.

## 7. Claim boundary

M8 proves an algebraic recurrence and conditional perturbation bounds for the
additive Ridge backend. It does not prove convergence of an optimizer, does
not transfer the Shampoo theorem, and does not guarantee that fixed INT8 has
width-independent accuracy. M6 remains a formal gate failure at width 20k.
