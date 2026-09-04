# Square-Root Quantized FLY: Structure-Preserving Analytic State Compression for Exemplar-Free Continual Learning

Working manuscript. The method and numbers in this draft correspond to the
locked Priority-2B (P2B) implementation. Claims remain provisional until the
literature review and an additional untouched evaluation are complete.

## Abstract

Exemplar-free analytic continual learners replace historical samples with
sufficient statistics, but their persistent state can remain large. In
FLY-style continual learning, a sparse random projection and winner-take-all
(WTA) operation expand a 768-dimensional frozen feature to a
10,000-dimensional code. Its dense float32 Ridge Gram alone contains 100
million entries. We present **Square-Root Quantized FLY (SRQ-FLY)**, which preserves the FLY
representation, code width, target statistic, global classifier, and
task-ID-free inference rule while replacing the dense Gram state with a
quantized triangular square root of the regularized system. The factor's
diagonal is stored in float32 and its strict upper triangle is stored with
deterministic groupwise int8 quantization. A blocked-QR update and streaming
factor encoder control update workspaces, while two triangular solves produce
the classifier. The reconstructed system retains the form
\(\widehat R^\top\widehat R\) and is positive definite whenever
\(\widehat R\) is nonsingular.

Across six paired class-order and projection replicates on CIFAR-100,
CUB-200-2011, and a disclosed legacy ImageNet-R split, P2B reduces measured
persistent learner state by 76.7--78.1% relative to width-10,000 Exact FLY.
The changes in average incremental accuracy are -0.018, -0.083, and -0.062
percentage points. In an isolated two-update Tesla T4 benchmark, P2B reduces
peak PyTorch CUDA allocation by 23.8%; in the real-data confirmation, analytic
update time remains 1.60--2.09 times that of Exact FLY. These findings support
SRQ-FLY as an accuracy--memory trade-off, not as an accuracy-improving
replacement for FLY.

## 1. Introduction

Class-incremental learning requires a single predictor over all classes seen
so far without task identity at inference. Replay can preserve historical
knowledge, but retaining images, embeddings, or codes makes learner state grow
with the stream. Analytic learners instead update a global classifier from
sufficient statistics and can avoid storing past samples under fixed-feature
assumptions.

Removing replay does not remove state memory. For stage features
\(X_t\in\mathbb R^{n_t\times m}\), a Ridge-style analytic learner maintains

\[
G_t=G_{t-1}+X_t^\top X_t,
\]

whose dense storage scales as \(O(m^2)\). This cost becomes important when the
representation is deliberately widened. FLY starts with a frozen
\(d=768\)-dimensional ViT feature, applies a fixed sparse projection, and
retains sample-dependent WTA activations in an \(m=10{,}000\)-dimensional
code. At this width, a float32 Gram requires 400,000,000 bytes (381.47 MiB)
before counting the projection, target statistic, classifier, or metadata.

Reducing \(m\) saves memory but changes the representation. Direct entrywise
quantization of the Gram need not preserve positive definiteness and can make
the Ridge solve fail. SRQ-FLY instead stores a square root of the regularized
system:

\[
H_t=G_t+\lambda I=R_t^\top R_t,
\]

where \(R_t\) is upper triangular with positive diagonal. It stores the
diagonal in float32 and quantizes only the strict upper triangle. The FLY code
that generates the analytic state remains unchanged.

This paper makes three contributions:

1. **A same-width compressed FLY learner.** SRQ-FLY replaces the dense
   width-10,000 Gram with a deterministic mixed-precision square-root state
   while preserving the upstream representation and inference semantics.
2. **Structure and error analysis.** We give explicit state-byte accounting,
   a structural positive-definiteness result, and bounds connecting factor
   error to the Ridge system, classifier, logits, and prediction margin.
3. **Audited accuracy--memory evidence.** We compare P2B with same-width Exact
   FLY, byte-matched lower-width Exact FLY, and raw-feature Ridge across three
   datasets and six paired replicates, while reporting persistent bytes, CUDA
   allocator peaks, numerical drift, analytic update time, and inference time
   separately.

We do not claim that SRQ-FLY improves FLY accuracy, compresses the backbone,
removes \(O(m^2)\) arithmetic, or already generalizes empirically to every
analytic continual learner.

## 2. Background and related work

### 2.1 Analytic continual learning

ACIL derives recursive analytic class-incremental learning without historical
data under fixed-feature assumptions [@zhuang2022acil]. F-OAL adapts recursive
least squares to forward-only online learning [@zhuang2024foal], while GACL
studies mixed exposed and unexposed classes [@zhuang2024gacl]. REAL, MoAL, and
AnaCP improve or adapt representations used by analytic classifiers
[@he2024real; @gao2025moal; @momeni2025anacp]. SRQ-FLY targets a complementary
bottleneck: the persistent quadratic state of a fixed, high-dimensional
analytic representation.

### 2.2 Random expansion and FLY

RanPAC combines a frozen pretrained representation with random nonlinear
expansion, prototype accumulation, and decorrelation [@mcdonnell2023ranpac].
FLY-CL uses a fly-inspired sparse projection and WTA code to reduce
multicollinearity while retaining an analytic update [@zou2026flycl]. Widening
can improve separation, but the corresponding dense second-order statistic
grows quadratically. SRQ-FLY preserves the repository's FLY representation and
changes only the storage and update representation of its regularized system.

### 2.3 Distribution summaries and quantized matrix state

FeCAM uses class-specific covariance geometry for classification
[@goswami2023fecam], whereas AdaGauss stores and adapts class distributions and
can generate pseudo-features [@rypesc2024adagauss]. SRQ-FLY neither models
per-class Gaussians nor generates samples; it compresses a global WTA-code
sufficient statistic.

Li et al. quantize Cholesky factors of Shampoo preconditioners and use error
feedback for stochastic optimization [@li2025fourbit]. That work motivates
factor-space quantization, but its optimizer, 4-bit format, error-feedback
state, inverse fourth root, and convergence theorem do not transfer directly
to streaming Ridge. SRQ-FLY uses groupwise int8, no error feedback, and a
continually updated analytic classifier.

Our novelty claim is deliberately narrow: SRQ-FLY adapts factor-space
quantization to the persistent regularized state of same-width FLY and couples
it with a blocked streaming update. A final submission must broaden the review
to classical square-root filtering, quantized recursive least squares, and
matrix sketching. The current ledger is not exhaustive.

## 3. Exact FLY and its state bottleneck

At stage \(t\), the learner receives current samples
\(\mathcal D_t=\{(x_i,y_i)\}_{i=1}^{n_t}\), where
\(x_i\in\mathbb R^d\) is a frozen-backbone feature and \(y_i\) is a global
class label. Let \(P\in\mathbb R^{m\times d}\) be the fixed sparse projection.
The code is

\[
z_i=\operatorname{WTA}_{\rho}(Px_i)\in\mathbb R^m.
\]

For code matrix \(Z_t\) and global one-hot target matrix \(Y_t\), Exact FLY
updates

\[
G_t=G_{t-1}+Z_t^\top Z_t,
\qquad
Q_t=\operatorname{expand}(Q_{t-1})+Z_t^\top Y_t.
\]

It solves

\[
H_tW_t=Q_t,
\qquad H_t=G_t+\lambda I,
\]

without forming an inverse. Prediction is
\(\arg\max_c z(x)^\top W_t[:,c]\), with no task identifier.

The dominant Exact-FLY tensor is the dense float32
\(G_t\in\mathbb R^{m\times m}\). Its payload is distinct from complete learner
state, which also includes the sparse projection, \(Q_t\), classifier, counts,
and metadata.

## 4. SRQ-FLY

### 4.1 Square-root recurrence

SRQ-FLY represents the regularized system with an upper-triangular factor:

\[
H_t=R_t^\top R_t.
\]

Because the previous factor has already been quantized, the implemented
recursion uses its decoded value \(\widehat R_{t-1}\). For later stages, the QR
factor of

\[
\begin{bmatrix}\widehat R_{t-1}\\Z_t\end{bmatrix}
\]

satisfies

\[
R_t^\top R_t
=\widehat R_{t-1}^\top\widehat R_{t-1}+Z_t^\top Z_t
\]

in exact arithmetic. SRQ-FLY therefore defines a deterministic approximate
recursion; after the first quantization it is not identical to Exact FLY's
historical Gram.

### 4.2 Final P2B update

A generic QR factorization would repeat work on the known triangular structure
of \(\widehat R_{t-1}\). P2B instead eliminates column panels with compact
Householder reflectors. With update rank \(n_t\), factor width \(m\), and panel
width \(p\), its leading work is proportional to \((n_t+p)m^2\), rather than
applying a generic cubic factorization to a dense \(m\times m\) system.

The locked backend uses:

- Gram--Cholesky for the first update;
- blocked QR for subsequent updates;
- panel size 128;
- block size 256 and quantization group size 64;
- streaming factor encoding in batches of 64 blocks.

An implicit-ridge QR initializer reduced isolated CUDA allocation further but
failed the preregistered real-data predictor-equivalence check. It is excluded
from the final method.

### 4.3 Mixed-precision factor storage

For a group \(r\) from the strict upper triangle, SRQ-FLY uses

\[
s(r)=\frac{\max_j|r_j|}{127},
\qquad
q_j=\operatorname{clip}_{[-127,127]}
\left(\operatorname{round}(r_j/s(r))\right).
\]

Zero groups use a finite positive unit scale. Each group stores one float32
scale and int8 payload values, and decoding gives
\(\widehat r_j=s(r)q_j\). The diagonal is copied in float32, preserving its
sign and avoiding coarse integer quantization at triangular-solve pivots.

The streaming encoder materializes floating-point values for only a bounded
batch of blocks, writes decoded values into the disposable factor used by the
solve, and releases temporary buffers before processing the next batch. This
changes allocation scheduling, not the quantization rule or checkpoint.

### 4.4 Classifier and state contract

The target statistic remains unquantized:

\[
Q_t=\operatorname{expand}(Q_{t-1})+Z_t^\top Y_t.
\]

The classifier follows from two triangular solves:

\[
\widehat R_t^\top V_t=Q_t,
\qquad
\widehat R_t\widehat W_t=V_t.
\]

The deployed state may contain the fixed projection, compressed factor,
float32 diagonal, scales, \(Q_t\), counts, class mapping, classifier, and
bounded metadata. It must not contain historical images, features, WTA codes,
per-sample labels, sample indices, or tensors indexed by historical samples.
Feature and WTA caches are experiment infrastructure, not learner state.

### 4.5 Algorithm

```text
Input: current codes Z_t, labels y_t, previous compressed factor, and Q
1. Expand Q to the global class set and add Z_t^T Y_t.
2. If t = 1, factor Z_t^T Z_t + lambda I by Cholesky.
3. Otherwise decode the previous factor and apply blocked QR to
   [R_hat_(t-1); Z_t].
4. Store the positive diagonal in FP32.
5. Quantize strict-upper blocks groupwise to INT8 in bounded batches.
6. Solve R_hat_t^T V_t = Q_t and R_hat_t W_hat_t = V_t.
7. Persist the compressed factor and permitted global state.
```

## 5. Memory and numerical analysis

### 5.1 Persistent-state complexity

For width \(m\), class count \(C\), group size \(g\), and sparse projection
with \(s\) nonzeros per row, dominant SRQ-FLY bytes are

\[
\Theta(ms)
+\frac{m(m-1)}{2}
+4m
+4\sum_{b\in\mathcal B}\left\lceil\frac{n_b}{g}\right\rceil
+8mC.
\]

The terms represent the sparse projection, int8 strict triangle, float32
diagonal, per-group scales, and float32 \(Q_t\) plus classifier. Exact FLY uses
a \(4m^2\)-byte float32 Gram instead of the factor terms.

At \(m=10{,}000\), the strict-upper int8 payload has 49,995,000 bytes and the
diagonal has 40,000 bytes before scales. The complete learner-state reduction
is smaller because projection and class-dependent tensors are shared. All
reported state figures sum actual persistent tensors.

### 5.2 Structural positive definiteness

**Proposition 1.** If \(\widehat R_t\) is upper triangular with nonzero
diagonal, then

\[
\widehat H_t=\widehat R_t^\top\widehat R_t\succ0.
\]

**Proof.** For any nonzero \(v\), nonsingularity gives
\(\widehat R_tv\neq0\), so
\(v^\top\widehat H_tv=\|\widehat R_tv\|_2^2>0\).

This exact-arithmetic structural result does not bound condition number,
floating-point residual, or prediction error.

### 5.3 Exact unquantized recurrence

**Proposition 2.** With \(R_0=\sqrt{\lambda}I\), unquantized QR updates satisfy

\[
R_t^\top R_t=\lambda I+\sum_{k=1}^{t}Z_k^\top Z_k.
\]

The proof follows by induction from the Gram identity of
\([R_{t-1};Z_t]\). Quantized SRQ-FLY uses \(\widehat R_{t-1}\) and is only
approximate.

### 5.4 Perturbation analysis

Let \(\widehat R=R+E\). The induced system perturbation is

\[
\Delta=\widehat R^\top\widehat R-R^\top R
=R^\top E+E^\top R+E^\top E,
\]

so

\[
\|\Delta\|_2
\le2\|R\|_2\|E\|_2+\|E\|_2^2.
\]

For \(W=H^{-1}Q\) and \(\widehat W=(H+\Delta)^{-1}Q\), assuming both systems
are positive definite,

\[
\|\widehat W-W\|_F
\le
\|H^{-1}\|_2\|\Delta\|_2
\|(H+\Delta)^{-1}\|_2\|Q\|_F.
\]

For code \(z\), logit error is at most
\(\|z\|_2\|\widehat W-W\|_2\). If the Exact-FLY top-class margin is
\(\gamma(z)\), the predicted class is preserved when maximum absolute logit
error is below \(\gamma(z)/2\).

These bounds explain sensitivity to quantization and spectral margin. They do
not establish that repeated error cannot accumulate; stage-wise drift remains
an empirical quantity pending a recurrence bound.

## 6. Experimental protocol

### 6.1 Datasets, streams, and selection

We use CIFAR-100, CUB-200-2011, and a legacy processed ImageNet-R split. The
locked protocol has 10 CIFAR tasks and 20 tasks for each 200-class dataset.
All methods use a frozen ViT-B/16 feature extractor with output dimension 768.

The final confirmation has six paired replicates with class-order seeds
3031--3036 and projection seeds 5031--5036. P2B and Exact FLY share width
10,000, synaptic degree 300, coding level 0.3, projection, WTA codes, task
split, and evaluation examples.

Hyperparameters were selected before confirmation with declared train-only
nested partitions. Locked Ridge values are:

| Dataset | Exact FLY/P2B \(\lambda\) | Raw Ridge \(\lambda\) |
|---|---:|---:|
| CIFAR-100 | \(10^6\) | 100 |
| CUB-200-2011 | \(10^5\) | 100 |
| ImageNet-R | \(10^6\) | 1000 |

The same test splits had been consumed by an earlier locked SRQ-FLY run. The
present results are backend-confirmation evidence, not untouched first-use
held-out evidence. No test metric changes a method, seed, or hyperparameter.

### 6.2 Baselines and metrics

The final comparison includes same-width Exact FLY, P2B, float64-statistics
raw-feature Ridge, and Exact FLY at a byte-matched lower width. The latter
width is derived without accuracy: 4,409 on CIFAR and 4,518 on both 200-class
datasets. A separate CIFAR train-only ablation includes a float16 factor and
direct int8 Gram.

Final accuracy is all-seen-class accuracy after the last stage. Average
incremental accuracy is

\[
\operatorname{AIA}=\frac1T\sum_{t=1}^{T}a_t,
\]

where \(a_t\) is all-seen-class accuracy after stage \(t\). We also report
forgetting, persistent tensor bytes, analytic update time, inference time,
solver residual, prediction agreement, and relative logit error.

Persistent bytes count deployed learner tensors. Peak allocated and reserved
CUDA bytes are separate PyTorch allocator measurements from isolated workers.
Disk caches and feature extraction are excluded from learner state and analytic
update time.

## 7. Main results

### 7.1 Fixed-width accuracy

Results are mean \(\pm\) sample standard deviation over six replicates. The
last column is a paired P2B-minus-Exact-FLY interval.

| Dataset | Final: Exact / P2B | AIA: Exact / P2B | \(\Delta\) AIA (pp), 95% CI |
|---|---:|---:|---:|
| CIFAR-100 | 88.632\(\pm\)0.138 / 88.580\(\pm\)0.106 | 92.249\(\pm\)0.447 / 92.231\(\pm\)0.420 | -0.018 [-0.076, 0.040] |
| CUB-200-2011 | 88.297\(\pm\)0.115 / 88.126\(\pm\)0.088 | 92.766\(\pm\)0.534 / 92.683\(\pm\)0.534 | -0.083 [-0.149, -0.017] |
| ImageNet-R (legacy) | 71.948\(\pm\)0.256 / 71.869\(\pm\)0.247 | 78.215\(\pm\)0.472 / 78.153\(\pm\)0.481 | -0.062 [-0.094, -0.030] |

P2B closely tracks but does not improve Exact FLY accuracy. The CIFAR interval
includes zero; the CUB and ImageNet-R intervals indicate a small negative mean
effect. The supported claim is preservation of most same-width FLY accuracy
under strong state compression, not statistical equivalence everywhere.

### 7.2 Persistent state

| Dataset | Exact FLY | P2B | P2B/Exact | Reduction |
|---|---:|---:|---:|---:|
| CIFAR-100 | 444.01 MB (423.44 MiB) | 97.17 MB (92.66 MiB) | 0.219 | 78.1% |
| CUB-200-2011 | 452.01 MB (431.07 MiB) | 105.17 MB (100.29 MiB) | 0.233 | 76.7% |
| ImageNet-R | 452.01 MB (431.07 MiB) | 105.17 MB (100.29 MiB) | 0.233 | 76.7% |

The 200-class state is larger because \(Q_t\) and the classifier grow with the
number of classes; P2B does not compress them.

### 7.3 Raw Ridge and predictor drift

Raw Ridge uses 5.95 MB on CIFAR and 7.18 MB on each 200-class dataset. Its mean
AIA is 91.441, 91.809, and 76.750, compared with P2B's 92.231, 92.683, and
78.153. P2B's mean final-accuracy advantages over Raw Ridge are 1.47, 2.33,
and 2.73 points. P2B is therefore an intermediate Pareto point rather than the
minimum-state solution.

Minimum stage-level prediction agreement between P2B and Exact FLY is 98.91%
on CIFAR, 98.42% on CUB, and 97.31% on ImageNet-R. Maximum relative logit
error is 0.218, 0.510, and 0.101. Close accuracy does not mean an identical
predictor. The maximum P2B solver relative residual over all final units is
\(9.77\times10^{-6}\). Mean forgetting for Exact/P2B is 4.293/4.194 on
CIFAR-100, 4.448/4.162 on CUB, and 6.549/6.477 on ImageNet-R; these descriptive
differences are not evidence that quantization improves forgetting.

## 8. State-matched and ablation evidence

### 8.1 Final state-matched Exact-FLY control

Shrinking Exact FLY's width is the simplest alternative way to meet P2B's
persistent-state budget. For each dataset, we choose the largest integer width
whose analytically computed Exact-FLY state does not exceed P2B state, then
select Ridge on the locked train-only nested partitions. The selected
configurations are:

| Dataset | Exact-FLY width | Selected \(\lambda\) | Exact/P2B state bytes | Relative byte gap |
|---|---:|---:|---:|---:|
| CIFAR-100 | 4,409 | \(10^6\) | 97,163,276 / 97,166,236 | 0.0030% |
| CUB-200-2011 | 4,518 | \(10^5\) | 105,149,848 / 105,166,636 | 0.0160% |
| ImageNet-R (legacy) | 4,518 | \(10^6\) | 105,149,848 / 105,166,636 | 0.0160% |

Six-replicate test results are:

| Dataset | Final: matched Exact / P2B | AIA: matched Exact / P2B | P2B-minus-matched AIA (pp), 95% CI |
|---|---:|---:|---:|
| CIFAR-100 | 87.915\(\pm\)0.112 / 88.580\(\pm\)0.106 | 91.767\(\pm\)0.396 / 92.231\(\pm\)0.420 | +0.464 [0.340, 0.589] |
| CUB-200-2011 | 87.856\(\pm\)0.137 / 88.126\(\pm\)0.088 | 92.552\(\pm\)0.566 / 92.683\(\pm\)0.534 | +0.132 [0.001, 0.262] |
| ImageNet-R (legacy) | 70.675\(\pm\)0.264 / 71.869\(\pm\)0.247 | 77.297\(\pm\)0.514 / 78.153\(\pm\)0.481 | +0.856 [0.698, 1.014] |

All three paired intervals lie above zero, although the CUB lower bound is
close to zero. This supports a specific Pareto claim: at nearly equal deployed
tensor bytes, preserving width 10,000 and compressing the square-root state is
more accurate than reducing Exact FLY to width 4,409--4,518. It does not imply
that quantization improves accuracy over width-10,000 Exact FLY; Section 7.1
shows the opposite small same-width effect on CUB and ImageNet-R.

### 8.2 Train-only component ablation

The locked CIFAR train-only ablation uses one deterministic development stream.

| Method | Validation AIA | Persistent state | Outcome |
|---|---:|---:|---|
| Exact FLY, width 10,000 | 92.257 | 444.01 MB | Complete |
| SRQ int8, width 10,000 | 92.270 | 97.17 MB | Complete |
| FP16 factor, width 10,000 | 92.260 | 144.04 MB | Complete |
| Exact FLY, width 4,409 | 91.720 | 97.16 MB | Complete |
| Raw Ridge | 91.175 | 2.97 MB | Complete |
| Direct int8 Gram | -- | -- | Non-positive-definite system at task 1 |

The final state-matched control confirms the development signal that retaining
width 10,000 while compressing state can be preferable to shrinking the
representation. The
direct-int8 failure shows that an unconstrained quantizer can destroy the
solve; it does not establish that all repaired direct quantizers must fail. A
separate Priority-3 train-only control is now preregistered and implemented,
but has not yet been executed. It propagates a Weyl lower bound from the
measured symmetric quantization-error infinity norm and applies a deterministic
diagonal loading without labels, validation accuracy, adaptive retries, or test
data. Until its artifact is audited, the paper must retain the narrow claim
above and must not attribute the Priority-1 outcome to square-root structure
alone.

## 9. System memory and runtime

### 9.1 Isolated CUDA benchmark

The P2B allocator study is synthetic-only: width 10,000, two updates, one
warm-up, seven measured repetitions, isolated Tesla T4 workers.

| Median quantity | Exact FLY | P2B | P2B/Exact |
|---|---:|---:|---:|
| Peak CUDA allocated | 1.570 GiB | 1.196 GiB | 0.762 |
| Peak CUDA reserved | 1.586 GiB | 1.246 GiB | 0.786 |
| Analytic time | 0.409 s | 0.762 s | 1.86 |

P2B reduces peak allocated memory by 23.8% and reserved memory by 21.4% in this
allocator-scoped benchmark. These are not whole-process NVML peaks and are not
substitutes for persistent-state bytes.

### 9.2 Real-data timing

| Dataset | Exact update | P2B update | Ratio | Exact inference | P2B inference |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 | 4.596 s | 9.594 s | 2.09 | 1.150 s | 1.116 s |
| CUB-200-2011 | 3.802 s | 6.068 s | 1.60 | 1.423 s | 1.392 s |
| ImageNet-R | 4.454 s | 8.281 s | 1.86 | 1.466 s | 1.416 s |

P2B remains slower during analytic updates because it decodes, applies blocked
QR, re-encodes the factor, and performs triangular solves. Inference semantics
are unchanged and measured inference time is similar. Shared feature
extraction is excluded.

## 10. Discussion and limitations

SRQ-FLY reduces FLY's persistent quadratic state without storing historical
samples or shrinking its WTA representation. Its factor form preserves an SPD
system structurally and exposes an explicit accuracy--memory trade-off.

The current limitations are:

- Storage and update arithmetic remain \(O(m^2)\).
- P2B does not compress the backbone, projection, \(Q_t\), classifier, or
  experiment caches.
- Real-data analytic update is 1.60--2.09 times slower than Exact FLY.
- The format is int8, not packed int4, and contains no error-feedback state.
- Structural positive definiteness does not guarantee conditioning or
  accuracy preservation.
- Final evidence covers one frozen ViT-B/16 and one FLY-style learner.
- Six replicates vary seeds but reuse the same dataset test samples.
- P2B confirms a backend on previously consumed test splits rather than an
  untouched held-out benchmark.
- The legacy ImageNet-R split has 19 cross-split duplicate hashes, including
  18 under conflicting labels, and cannot be called content-disjoint.
- The current state-matched archive is recovery evidence rather than a fully
  source-locked final artifact. Its test-feature extraction used an in-memory
  compatibility adapter that ordered the repository's `{task_id: DataLoader}`
  dictionary after the locked runner incorrectly iterated dictionary keys.
  The adapter changed no sample, model, seed, width, or hyperparameter, but it
  was not part of the original authorization source identity. A clean rerun on
  the corrected runner is required before treating this control as final paper
  evidence.

The algebra may apply when another learner's deployed state contains an SPD
regularized second-order matrix. This remains a hypothesis until an independent
analytic learner is implemented and audited.

## 11. Conclusion

Exemplar-free analytic learning removes historical samples but can retain a
large quadratic state. SRQ-FLY compresses the regularized FLY system through a
mixed-precision triangular factor while preserving same-width FLY features.
The locked P2B implementation reduces persistent state by 76.7--78.1% and
isolated peak CUDA allocation by 23.8%, while changing mean AIA by at most
0.083 points across the three evaluated datasets. At nearly the same state
budget, it improves AIA over lower-width Exact FLY by 0.132--0.856 points.
This comes with a 1.60--2.09 times analytic-update overhead. The evidence
supports SRQ-FLY as structure-preserving state compression that avoids the
accuracy cost of shrinking representation width, not as an accuracy
improvement over same-width FLY or a universal solution for analytic
continual learning.

## Evidence provenance

- Final artifact: `srq_fly_p2b_final_confirmation.zip`, SHA-256
  `14826488b8d82bc306a07e6d4f229cc389a8447150833aefc1de664961a9e85d`.
- Locked implementation commit:
  `86a9e8f242925d5c50d1ab251088e6fbb9e2944a`.
- Final status: `CONFIRMATION_REPORTED_WITHOUT_ACCURACY_GATE`.
- System artifact: `srq_fly_priority2b_memory.zip`.
- Development ablation: `srq_fly_priority1_train_only.zip`.
- State-matched artifact: `srq_fly_state_matched_final.zip`, SHA-256
  `a5adc883089f6108a01f33d57f0737894af843262a18a50f5309d82a54f323f9`.
- State-matched train-only checkpoint:
  `srq_state_matched_train_only_checkpoint.zip`, SHA-256
  `9c42d3f51581443b642b8b79e793d44f412a73936fc8e45cf9cd7238dcb22801`.
- State-matched run commit:
  `ccd211c3d3f1c5ac5e3855431bdfeba69708b422` (with the extraction-adapter
  caveat above).

## References

Citation metadata and source URLs are recorded in
[`references.bib`](references.bib) and
[`RELATED_WORK_LEDGER.md`](RELATED_WORK_LEDGER.md).
