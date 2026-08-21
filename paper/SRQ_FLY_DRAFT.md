# Square-Root Quantized FLY for Memory-Efficient Exemplar-Free Analytic Continual Learning

Working manuscript. All real-data numbers currently come from train-only
validation protocols. The title, abstract, and claims are provisional.

## Abstract

Analytic class-incremental learners can update a global classifier from
sufficient statistics without replay, but the quadratic state of a
high-dimensional expanded representation can dominate deployment memory. We
study SRQ-FLY, an exemplar-free and task-ID-free variant of FLY-style analytic
continual learning that replaces the dense expanded-space Gram matrix with a
groupwise-int8 triangular square-root state. The diagonal is retained
unquantized in float32, so the reconstructed Ridge system is positive definite by
construction and classifier updates use triangular solves rather than an
explicit inverse. The learner checkpoint contains no historical image,
feature, code, per-sample label vector, or sample index. In locked ImageNet-R and CUB
train-validation studies, SRQ-FLY uses approximately 23.27% of the persistent
tensor state of exact FLY at expansion width 10,000 while closely tracking its
accuracy. Against an exact FLY representation selected to match persistent
state, the observed gain is positive on ImageNet-R and averages 0.515
percentage point over five additional CUB seeds. This CUB interval includes zero,
and one preregistered predictor-agreement gate fails narrowly; therefore these
results establish a promising accuracy-memory signal rather than a held-out or
statistically significant superiority claim.

## 1. Introduction

Class-incremental learning requires a single predictor over all classes seen
so far, without knowing task identity at inference. Replay-based methods can
retain past information effectively, but storing images or per-example
features makes the learner state grow with the stream and weakens a deployable
exemplar-free claim. Analytic learners instead preserve matrix sufficient
statistics and recompute a deterministic global classifier after each task.

FLY-style representations apply a fixed sparse expansion followed by a
sample-dependent winner-take-all operation. Their high-dimensional sparse
codes can be accurate, but an exact Ridge Gram matrix grows quadratically with
the expansion dimension. Reducing that dimension saves memory while changing
the representation itself. Directly quantizing the Gram matrix is another
option, but quantization can destroy positive definiteness and make the
analytic solve unreliable.

SRQ-FLY changes the storage representation of the Ridge system rather than the
FLY code dimension. It streams a triangular square root, stores its strict
triangle using deterministic groupwise int8 quantization, preserves the
diagonal without integer quantization, and solves the classifier through two
triangular systems.
The method is deliberately simple: the frozen backbone, sparse projection,
winner-take-all code, target statistic, and global inference rule remain
unchanged.

The current contributions are:

1. a square-root streaming formulation for an expanded-space analytic
   continual classifier whose reconstructed system is positive definite by
   construction;
2. a deterministic compact checkpoint using an unquantized float32 diagonal and
   groupwise-int8 strict-triangle storage, with explicit byte accounting and
   no sample-level learner state;
3. state-matched controls separating the effect of compressed sufficient
   statistics from simply reducing FLY expansion width;
4. an audit-driven evaluation that reports negative gates and distinguishes
   experiment caches, runtime memory, and persistent learner state.

No contribution currently claims held-out superiority, an int8 backbone,
error feedback, or train-from-scratch continual representation learning.

## 2. Related work

**Analytic continual learning.** ACIL derives an exemplar-free recursive
analytic learner whose solution is equivalent to its joint-learning
counterpart under the paper's fixed-feature assumptions [@zhuang2022acil].
F-OAL extends recursive least squares to an online, forward-only setting with
a frozen encoder [@zhuang2024foal], while GACL addresses generalized streams
with mixed exposed and unexposed classes [@zhuang2024gacl]. More recently,
REAL enhances representations before recursive analytic classification
[@he2024real], MoAL revisits adaptation of analytic learners with pretrained
models [@gao2025moal], and AnaCP adds an analytic contrastive projection
without gradient-based continual updates [@momeni2025anacp]. SRQ-FLY addresses
a different bottleneck: it keeps the FLY representation fixed and compresses
the quadratic sufficient state used by its analytic classifier.

**Pretrained representations and random expansion.** RanPAC combines a frozen
pretrained representation, random nonlinear expansion, prototype
accumulation, and prototype decorrelation [@mcdonnell2023ranpac]. FLY-CL uses a
fly-inspired sparse expansion to reduce multicollinearity and training cost in
pretrained-model continual learning [@zou2026flycl]. SRQ-FLY does not propose a
new expansion or winner-take-all rule; it preserves the repository's FLY code
and changes only the storage and update representation of the regularized
analytic system.

**Exemplar-free distribution summaries.** FeCAM models heterogeneous class
distributions through covariance-aware distances [@goswami2023fecam].
AdaGauss adapts class covariances and can replay pseudo-features from Gaussian
class summaries when the feature extractor changes [@rypesc2024adagauss].
Those methods use class-distribution geometry for classification or synthetic
replay. SRQ-FLY instead stores global WTA-code sufficient statistics and never
generates pseudo-samples.

**Quantized matrix state.** Li et al. quantize Cholesky factors of Shampoo
preconditioners and add error feedback for stochastic optimization
[@li2025fourbit]. That work motivates factor-space quantization as a matrix
compression pattern, but its optimizer, 4-bit quantizer, error state, and
convergence analysis do not transfer directly to streaming Ridge. SRQ-FLY uses
groupwise int8 storage, no error feedback, and proves only the algebraic
properties stated in Section 5.

The repository's current SOHO path is treated as a local experimental method,
not as an externally established publication: it combines OLDA, ETF or
Procrustes alignment, dynamic WTA, and historical-feature re-projection. No
authoritative SOHO paper artifact is present in this repository, so the final
manuscript must not attach an external novelty or priority claim to SOHO until
the source report is supplied and audited.

## 3. Problem formulation

At stage \(t\), the learner observes only the current training batch

\[
\mathcal D_t=\{(x_i,y_i)\}_{i=1}^{n_t},
\]

where \(x_i\in\mathbb R^d\) is a frozen-backbone feature and \(y_i\) is a
global class label. Past examples and past per-example features or codes are
unavailable. Inference receives an individual feature but no task identifier
and must classify over all classes seen through stage (t).

Let \(H\in\mathbb R^{m\times d}\) be the fixed sparse FLY projection. The code
for one feature is

\[
z_i=\operatorname{WTA}_{\rho}(Hx_i)\in\mathbb R^m,
\]

where the sample-dependent operation retains the largest signed activations
under the locked coding level \(\rho\). For a current-stage code matrix
\(Z_t\in\mathbb R^{n_t\times m}\) and a global one-hot target matrix
\(Y_t\in\mathbb R^{n_t\times C_t}\), exact analytic FLY maintains

\[
G_t=G_{t-1}+Z_t^\top Z_t,
\qquad
Q_t=\operatorname{expand}(Q_{t-1})+Z_t^\top Y_t.
\]

The global Ridge classifier is

\[
W_t=(G_t+\lambda I)^{-1}Q_t,
\qquad
\ell(x)=z(x)^\top W_t.
\]

The implementation never forms the inverse explicitly.

## 4. SRQ-FLY

### 4.1 Streaming square-root state

Define the regularized system

\[
A_t=G_t+\lambda I=R_t^\top R_t,
\]

where \(R_t\) is upper triangular with positive diagonal. Given a decoded
previous factor \(\widetilde R_{t-1}\), the implemented update forms

\[
B_t=\widetilde R_{t-1}^\top\widetilde R_{t-1}+Z_t^\top Z_t
\]

for \(t>1\), and includes \(\lambda I\) at initialization. A Cholesky
factorization produces the next exact local factor before storage:

\[
R_t=\operatorname{chol}(B_t)^\top.
\]

This Cholesky form is equivalent to a QR square-root update in exact
arithmetic. Under quantized storage, it defines the actual deterministic SRQ
recursion rather than an exact reconstruction of the uncompressed historical
Gram.

### 4.2 Deterministic groupwise quantization

The diagonal of \(R_t\) is stored without integer quantization in float32.
Under the locked float32 solver this is a direct copy of the local Cholesky
diagonal. Strict-upper values are partitioned by matrix blocks and then into
groups of size \(g\). For group
\(v\), the symmetric scale and integer payload are

\[
s(v)=\frac{\max_j|v_j|}{127},
\qquad
q_j=\operatorname{clip}_{[-127,127]}
\left(\operatorname{round}(v_j/s(v))\right).
\]

Zero groups use a finite positive unit scale. Quantization is deterministic;
each group stores one float32 scale and int8 values. Decoding restores
\(\widetilde v_j=s(v)q_j\) and combines the strict triangle with the exact
positive diagonal.

### 4.3 Analytic classifier update

After expanding the class columns of \(Q_{t-1}\), SRQ updates \(Q_t\) without
SRQ quantization in the chosen statistics dtype. It then solves

\[
\widetilde R_t^\top U_t=Q_t,
\qquad
\widetilde R_tW_t=U_t.
\]

The classifier is global over all seen classes. Neither update nor inference
requires a task identifier.

### 4.4 Persistent state and exemplar-free contract

Allowed checkpoint tensors are:

- the fixed sparse projection \(H\);
- the compressed triangular factor, unquantized diagonal, and group scales;
- \(Q_t\), class counts and class mapping;
- the derived global classifier \(W_t\);
- bounded scalar configuration and audit metadata.

Historical images, backbone features, WTA codes, per-sample label histories,
sample indices, or any tensor carrying a historical-sample dimension are
forbidden. Feature and
WTA caches used to accelerate experiments are sample-level disk
infrastructure and must not be packaged in the learner checkpoint.

### 4.5 State complexity

For expansion width \(m\), class count \(C\), group size \(g\), feature
dimension \(d\), and \(s\) nonzeros per projection row, the dominant bytes are

\[
\underbrace{\Theta(ms)}_{\text{sparse projection}}
+\underbrace{\tfrac{m(m-1)}{2}}_{\text{int8 strict triangle}}
+\underbrace{4m}_{\text{unquantized diagonal}}
+\underbrace{4\sum_{b\in\mathcal B}
\left\lceil\tfrac{n_b}{g}\right\rceil}_{\text{per-block scales}}
+\underbrace{8mC}_{Q_t\text{ and }W_t},
\]

where \(\mathcal B\) is the set of stored upper-triangular blocks and \(n_b\)
is the number of strict-upper payload entries in block \(b\). The sum is
blockwise because quantization groups restart at every stored block. The
remaining terms include counts and sparse CSC index metadata. Exact FLY
instead stores a dense float32 \(m\times m\) Gram term. All reported state
comparisons use measured runtime tensors rather than this asymptotic
expression alone.

## 5. Analysis

### Proposition 1: structural positive definiteness

If the decoded upper-triangular factor \(\widetilde R_t\) has strictly positive
diagonal, then

\[
\widetilde A_t=\widetilde R_t^\top\widetilde R_t\succ0.
\]

Therefore both triangular solves are well-defined in exact arithmetic. This
is a structural numerical property, not an accuracy guarantee.

### Proposition 2: exact square-root streaming equivalence

Without quantization and with fixed \(\lambda>0\), initializing
\(R_0^\top R_0=\lambda I\) and applying the square-root update yields

\[
R_t^\top R_t=\lambda I+\sum_{k=1}^t Z_k^\top Z_k.
\]

The proof follows by induction. The quantized learner follows an approximate
recursion and must not be described as exact streaming Ridge.

### Proposition 3: classifier perturbation

Let \(A_t=G_t+\lambda I\), \(\widetilde A_t=A_t+\Delta_t\), and assume both
are positive definite. With \(W_t=A_t^{-1}Q_t\) and
\(\widetilde W_t=\widetilde A_t^{-1}Q_t\), the resolvent identity gives

\[
\|\widetilde W_t-W_t\|_F
\le
\|A_t^{-1}\|_2\,\|\Delta_t\|_2\,
\|\widetilde A_t^{-1}\|_2\,\|Q_t\|_F.
\]

For a code \(z\), logit perturbation is bounded by

\[
\|z^\top(\widetilde W_t-W_t)\|_2
\le \|z\|_2\|\widetilde W_t-W_t\|_2.
\]

These bounds explain dependence on quantization error and the Ridge spectral
margin; they do not predict an accuracy improvement.

## 6. Experimental protocol

All compared methods within a run share the frozen ViT checkpoint,
preprocessing, class order, task split, seed, and evaluation examples.
Hyperparameters are selected only on declared inner training-validation
partitions. The outer train-validation fold is evaluated once after locking.
Held-out test features remain absent.

The mandatory controls are exact FLY at width 10,000, exact FLY at width 4,518
chosen by state accounting, float64 streaming raw-feature Ridge, and SRQ-FLY
at width 10,000. Existing SOHO/FLY paper comparisons require a separate audit
when their checkpoints retain sample-level caches or replay features.

Metrics include accuracy after every stage, final accuracy, average
incremental accuracy, forgetting when well-defined, update/inference time,
peak runtime memory, and persistent learner-state bytes. Disk caches are
reported separately.

## 7. Current train-validation evidence

### 7.1 ImageNet-R state-matched control

| Method | Validation AA | Final accuracy | Persistent state |
|---|---:|---:|---:|
| SRQ-FLY-10000 | **77.9343** | **71.1197** | 105,166,628 B |
| Exact FLY-4518 | 77.0141 | 70.0085 | 105,149,848 B |

The observed state-matched differences are +0.9201 average and +1.1112 final
points. This is one seed and not a held-out result.

### 7.2 CUB single-seed replication

| Method | Validation AA | Final accuracy | Persistent state |
|---|---:|---:|---:|
| SRQ-FLY-10000 | **91.7564** | **87.5282** | 105,166,628 B |
| Exact FLY-10000 | 91.6761 | 87.1102 | 452,006,940 B |
| Exact FLY-4518 | 91.5147 | 86.9393 | 105,149,848 B |
| Raw Ridge | 89.6107 | 84.8446 | 7,177,792 B |

D3 formally stopped because two rejected inner candidates narrowly exceeded
its universal numerical threshold. Selected and outer models were stable.

### 7.3 CUB five-seed replication

| Method | Mean validation AA | Mean final accuracy | Persistent state |
|---|---:|---:|---:|
| Exact FLY-10000 | **91.6070** | **87.2753** | about 452.007 MB |
| SRQ-FLY-10000 | 91.5699 | 87.1086 | about 105.167 MB |
| Exact FLY-4518 | 91.0547 | 86.5035 | about 105.150 MB |
| Raw Ridge | 90.0239 | 83.9767 | about 7.178 MB |

SRQ gains +0.5153 average and +0.6051 final points over state-matched FLY and
wins average accuracy on four of five seeds. The 95% t interval for the
average gain is [-0.1162, +1.1467], so it includes zero. D4 formally stopped
because one task/seed prediction-agreement value was 97.8178% versus the
preregistered 98% minimum.

## 8. Limitations and open evidence

The real-data evidence is train-validation only. The CUB multi-seed interval
does not establish a statistically nonzero population gain. Predictor fidelity
degrades slightly late in some streams even when accuracy remains close to
exact FLY. The backbone is frozen and full-precision; no claim concerns
train-from-scratch continual learning or end-to-end quantization. Experiment
caches contain sample-level data and cannot be shipped as exemplar-free
learner state. Runtime from heterogeneous Colab environments is not directly
comparable with paper hardware.

Held-out CIFAR-100 and CUB evaluation, plus legacy processed-split ImageNet-R
evaluation, requires a separately committed, single-use protocol. A raw-byte
identity audit of the current processed
ImageNet-R artifact found 19 content hashes crossing train/test, including 18
under conflicting class directories. The project retains that split only for
a fully disclosed legacy comparison; it cannot support a content-disjoint or
untouched-held-out claim.
Error feedback and lower-bit storage are deferred and are not contributions
of the current method.

## 9. Conclusion

SRQ-FLY provides a structurally positive-definite compressed sufficient state
for exemplar-free analytic continual learning. Existing train-validation
evidence suggests that it closely tracks a much larger exact FLY classifier
and can outperform an exact FLY representation at matched persistent memory.
The evidence is promising but not yet a held-out or statistically conclusive
paper result.

## References

Citation metadata and primary-source URLs are recorded in
[`references.bib`](references.bib) and
[`RELATED_WORK_LEDGER.md`](RELATED_WORK_LEDGER.md).
