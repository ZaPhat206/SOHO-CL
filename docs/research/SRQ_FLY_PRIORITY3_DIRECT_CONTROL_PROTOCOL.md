# SRQ-FLY Priority 3: direct-quantization control

## Question

This train-only phase asks whether SRQ-FLY succeeds because it stores a
square-root factor, or merely because it uses fewer bits. Priority 1 already
showed that naive groupwise-int8 quantization of the Gram matrix can make the
Ridge system non-positive-definite. That failure alone is not a complete
control: a reviewer can reasonably ask whether a principled repair would make
direct Gram quantization competitive.

No command in this phase loads `test.pt`, and no held-out evaluation is
authorized by completion of this study.

## Paired methods

All five rows use the same frozen ViT-B/16 training features, seed 2025,
class order, train-validation split, width 10,000, sparse projection, WTA code,
and FLY Ridge lambda (`1e6`):

1. `exact_fly_10000`: dense float32 Gram reference;
2. `direct_int8_gram_naive`: direct groupwise-int8 Gram with no repair;
3. `direct_int8_gram_weyl_repair`: the same direct-int8 Gram plus the locked
   certificate below;
4. `sqrt_float16`: upper Cholesky factor with float32 diagonal and float16
   strict-upper entries;
5. `srq_int8_p2b`: the current mixed INT8/FP32 SRQ factor with the locked P2B
   blocked-QR and streaming-quantization backend.

Exact FLY is an anchor, not another low-bit candidate. Naive direct INT8 is
allowed to return its already-observed numerical failure. The other four rows
must complete and satisfy the same solver-residual tolerance. Accuracy does
not determine whether the phase is considered methodologically complete.

## Locked direct-Gram repair

Let \(\widehat G_{t-1}\) denote the decoded direct-int8 Gram state and let

\[
S_t=\widehat G_{t-1}+Z_t^\top Z_t,
\qquad
\widehat G_t=Q_8(S_t)=S_t+E_t.
\]

The runner propagates a scalar certified lower bound. Starting from
\(\ell_0=0\), it computes

\[
\epsilon_t=\lVert E_t\rVert_\infty,
\qquad
\ell_t=\ell_{t-1}-\epsilon_t.
\]

Each update adds a positive-semidefinite matrix. Moreover, `S_t`, `E_t`, and
the reconstructed state are explicitly symmetric. For symmetric \(E_t\),

\[
\lVert E_t\rVert_2
\leq\sqrt{\lVert E_t\rVert_1\lVert E_t\rVert_\infty}
=\lVert E_t\rVert_\infty.
\]

Weyl's inequality therefore gives
\(\lambda_{\min}(\widehat G_t)\geq\ell_t\). The solve uses

\[
\widehat G_t+(\lambda+\delta_t)I,
\qquad
\delta_t=\max\{0,\mu_t-(\lambda+\ell_t)\},
\]

where the floating-point safety margin is fixed before the run as

\[
\mu_t=8\,\epsilon_{\mathrm{fp32}}\,m\,
\max\{\lVert\operatorname{diag}(\widehat G_t)\rVert_\infty,\lambda,1\}.
\]

Thus the certified system floor is positive in the stated arithmetic model.
The repair never reads labels beyond the ordinary Ridge cross-statistic,
validation accuracy, test data, or eigensolver output. There is no retry loop
and no post-failure jitter. Its two scalar certificate values are included in
persistent-state accounting.

This certificate is intentionally conservative. If it needs a large loading
and loses accuracy, that is an interpretable negative result for direct Gram
quantization, not permission to tune the repair after seeing validation.

## Interpretation locked before execution

Let \(d\) be validation AIA of `srq_int8_p2b` minus validation AIA of the
repaired direct control:

- \(|d|\leq0.1\) pp: low-bit storage plus certified repair is sufficient on
  this development stream; a square-root-specific accuracy claim is not
  supported;
- \(d\geq0.5\) pp: material square-root accuracy advantage;
- \(0.1<d<0.5\) pp: modest square-root accuracy advantage;
- \(d<-0.1\) pp: repaired direct Gram is better on this development stream.

These labels summarize one paired train-validation experiment. They are not
confidence intervals and are not paper-level held-out conclusions.

## Measurements and stop rule

Each method runs in a fresh process. The artifact reports validation AIA,
per-task accuracy, update time, persistent tensor bytes, peak PyTorch CUDA
allocated/reserved bytes, and solver residual. The repaired control also
reports its local quantization-error bound, certified Gram lower bound,
diagonal loading, effective Ridge lambda, and certified system floor per task.

The phase stops if a non-naive method fails, a repair certificate is missing,
the solver tolerance exceeds `2e-5`, or `test.pt` becomes visible. A completed
artifact is returned for audit before any report claim is changed.
