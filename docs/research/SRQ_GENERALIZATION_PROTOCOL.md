# SRQ analytic-backend generalization protocol

Status: active staged protocol. This document does not authorize a held-out
test run. A later milestone may begin only after the preceding gate has a
recorded PASS result.

## Research question

The study asks whether structure-preserving square-root compression is a
reusable backend for high-dimensional analytic continual learners, rather
than an optimization specific to FLY. The admissible learner family has a
fixed explicit feature map for historical samples and additive Ridge
sufficient statistics

\[
A_t=\Lambda+\sum_{k\le t}\Phi_k^\top\Phi_k,\qquad
B_t=\sum_{k\le t}\Phi_k^\top Y_k,\qquad
W_t=A_t^{-1}B_t.
\]

The protocol does not claim compatibility with a changing historical feature
map, prototype-only classifiers, iterative softmax training, covariance
downdates, negative weights, or inverse-RLS implementations without a
separate mathematically equivalent adapter.

## Frozen current method

Until the FLY regression gate passes, the reference compressed method is P2B:

- mixed INT8/FP32 upper square-root storage;
- FP32 diagonal and groupwise-INT8 strict upper triangle;
- block size 256 and group size 64;
- blocked QR with panel size 128;
- first update by Gram--Cholesky;
- streaming quantization in batches of 64 blocks.

No precision allocation, error feedback, packed INT4, Ridge retuning, or
checkpoint-format change is allowed during backend extraction.

## Comparison contract

Within one comparison, every backend receives byte-identical `Phi` and labels
and uses the same class order, task grouping, Ridge coefficient, dtype, seed,
and evaluation schedule. Method-specific hyperparameters are selected using
train-only validation. Test accuracy cannot choose a backend, threshold,
width, rank, precision policy, or stopping point.

Persistent bytes include the projection when it is deployed with the
frontend, the quadratic state, quantization scales and metadata tensors,
cross statistic, class counts, and classifier. Feature/code caches are
reported separately and are never treated as learner state.

## Milestones and gates

### M0 -- evidence and source lock

Work:

1. record the current source, config, artifact, test-use, and caveat identities;
2. exclude logs and local experiment caches from source control;
3. synchronize the manuscript results ledger with the immutable evidence;
4. run the focused SRQ regression suite and the full repository suite.

Gate: the machine-readable manifest parses, every referenced repository file
exists, document consistency checks pass, and the exact test results are
recorded. A dirty unrelated worktree is allowed for local documentation only;
all experiment workers must later run from a clean committed clone.

### M1 -- generic backend extraction

Work: extract class expansion, cross-statistic update, Exact Gram, square-root
update, compressed storage, state accounting, checkpointing, and diagnostics
under `methods/analytic_ridge/`. FLY projection and WTA remain in a frontend
wrapper.

Gate: legacy and extracted P2B produce byte-identical compressed blocks,
classifier tensors within the locked numerical tolerance, identical
persistent bytes, and compatible checkpoint round trips on synthetic streams.

Required tests (once created):

```text
python -m pytest -q tests/test_analytic_ridge_backend.py
```

### M2 -- unquantized equivalence

Work: compare Exact Gram, dense QR, and blocked QR in FP64 and FP32. Record
factor/system error, weight error, solver residual, logit drift, and prediction
agreement after every task.

Gate: FP64 blocked QR matches dense QR and Exact Gram at numerical tolerance;
FP32 square-root introduces no material prediction or accuracy change on the
declared train-only development stream.

### M3 -- FLY train-only regression (PASS)

Work: compare legacy and generic Exact/P2B learners on the same cached CIFAR
training stream.

Gate: representation, target statistics, selected hyperparameters, persistent
bytes, and predictions satisfy the preregistered regression tolerances. This
is the earliest milestone that may require a Colab GPU run.

Recorded artifact: `srq_generalization_m3_fly_regression.zip`, SHA-256
`ec376875ff0e29e2d7a46d1fcb9fcccc523ea47fbe6a8f2dd83ec3a98930a595`.
All ten task records pass exact legacy/generic tensor and byte identity.

### M4 -- RanPAC adapter and train-only gate (PASS)

Work: reproduce the original analytic RanPAC head, then compare original,
generic Exact Gram, FP32 square-root, FP16 square-root, and fixed P2B SRQ on
identical expanded features.

Gate: original and generic Exact semantics match; FP32 square-root matches
Exact; fixed SRQ reduces the quadratic persistent state by at least 70%, has
at most 0.20 percentage-point validation-AIA loss, and does not show a
numerical failure. Failure keeps the paper scoped to SRQ-FLY and cannot be
overridden using test accuracy.

The source-locked run instructions and the distinction between the official
per-task Ridge schedule and the controlled fixed-Ridge comparison are in
`docs/research/SRQ_GENERALIZATION_M4_RUNBOOK.md`.

Recorded artifact: `srq_generalization_m4_ranpac_train_only.zip`, SHA-256
`228da0828c7f6964bcc8f7da92258efa20b00eef0fc83679246ce0ae2a0c8f52`.
All eight gates passed on the locked CIFAR train-only stream. This closes the
initial non-FLY frontend gate but remains one controlled development seed.

### M5 -- equal-budget alternatives (PASS)

Work: compare full-width Exact, byte-matched reduced width, FP16 square-root,
fixed SRQ, raw-feature Ridge, and a fixed signed-hash feature-sketch Ridge.
Sketch dimensions and frontend widths are derived from total persistent byte
budgets before accuracy is read. The source-locked procedure is specified in
`docs/research/SRQ_GENERALIZATION_M5_RUNBOOK.md`.
The previously preregistered low-rank/streaming Ridge sketch slot is therefore
instantiated by an explicit CountSketch feature map followed by Exact Ridge;
it is not mislabeled as a factor-space approximation of the full Gram.

Gate: SRQ lies on or close to the observed accuracy--persistent-state Pareto
frontier. If a sketch dominates SRQ in accuracy, state, and update cost, the
claim must be narrowed before proceeding.

Recorded artifact: `srq_generalization_m5_equal_budget_train_only.zip`,
SHA-256
`0f5e23fa4a7d83926638641025fb103895561073cd1f0fa4e1e0d552fd2fa931`.
All seven gates passed. At approximately 91.9 MB total persistent state, P2B
exceeded reduced-width Exact and CountSketch Exact by 0.4838 and 0.5720
validation-AIA points, respectively. P2B was not Pareto dominated by the five
tested alternatives, but its analytic update was 15.25 and 19.22 times slower
than those two equal-budget controls. This remains one train-only development
seed and does not establish a global Pareto frontier.

### M6 -- width scaling (formal FAIL; diagnostic complete)

Work: sweep fixed widths 2k, 4k, 6k, 8k, 10k, 15k, and 20k for Exact Gram, FP16
square-root, and fixed P2B on the controlled RanPAC train-only stream. Widths
are reporting points, not accuracy-selected candidates. Projection prefixes,
splits, and per-width Ridge calibration are shared across the three backends.
The source-locked procedure and plot contract are in
`docs/research/SRQ_GENERALIZATION_M6_RUNBOOK.md`.

Gate: expose the empirical accuracy--state--update curves, preserve numerical
and accuracy-retention bounds at every width, and retain the quadratic state
terms in the report.

Recorded artifact: `srq_generalization_m6_width_sweep_train_only.zip`,
SHA-256
`b2739b9da023ebd2eedb6fdfe01c394e94f252773e847533b35350021c3d239e`.
The artifact is complete, train-only, source-clean, and numerically stable,
but its formal status is `FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY`. At width 20,000,
P2B loses `0.255845` validation-AIA points to Exact, exceeding the locked
`0.25`-point bound by `0.005845` points. The threshold is not changed after
observing the result. FP16 loses at most `0.0100` points and the maximum solver
residual is `5.83e-6`, which localizes the observed degradation to INT8 state
approximation rather than failure of the square-root update or linear solve.
M7 is therefore a diagnostic follow-up, not a retroactive M6 rescue.

### M7 -- task-wise error trajectory (PASS)

Work: measure task-wise factor/system/solution/logit error, prediction
agreement, and margin diagnostics at widths 10,000 and 20,000. These are
predeclared diagnostic points, not candidates for selecting a replacement
width or precision from validation accuracy.

Recorded artifact: `srq_generalization_m7_error_trajectory_train_only.zip`,
SHA-256
`df92adadce046c53efa5b9fcf01435d1fab2a4c71a690b10c3d7c221205acf36`.
All integrity and numerical gates passed. At task 10, the P2B system-action,
weight, and logit relative errors are `0.00859 / 0.0272 / 0.3178` at width
10,000 and `0.00902 / 0.0488 / 0.6351` at width 20,000. Prediction agreement
falls from `98.46%` to `97.24%`, while the sufficient margin-certificate rate
falls from `84.19%` to `75.43%`. FP16 remains close to Exact. The result is
consistent with stronger downstream amplification of INT8 perturbation at
larger width, but the one-seed probe correlations are not a causal or
condition-number estimate.

### M8 -- perturbation theory (COMPLETE)

Work: state the exact square-root identity, structural-SPD result, cumulative
factor-to-system error recurrence, Ridge-solution perturbation bound, and a
margin-preservation corollary. Every assumption must be explicit, and no
optimizer-convergence theorem may be transferred from Shampoo.

Recorded derivation: `docs/research/SRQ_GENERALIZATION_M8_THEORY.md`. The
manuscript now contains the exact recurrence
`Delta_t = Delta_{t-1} + D_t`, its spectral-norm accumulation bound, the
conditional Ridge-solution bound under `||A_t^{-1} Delta_t||_2 < 1`, and the
sufficient top-1 margin certificate. The text explicitly distinguishes these
theoretical quantities from M7's randomized diagnostics.

Gate: each claimed bound follows from the stated recurrence, distinguishes
local quantization error from cumulative effective-system error, and matches
the quantities measured in M7.

### M9 -- repeated systems evidence (PASS)

Work: run at least three isolated systems repetitions separating persistent
bytes, checkpoint bytes, PyTorch allocated/reserved peaks, process NVML peak,
and per-stage time.

Implementation: four paired repetitions are locked in
`configs/srq_generalization_m9_repeated_systems_train_only.json`, with balanced
Exact/SRQ execution order and eight fresh whole-process workers. The notebook
is `notebooks/srq_generalization_m9_repeated_systems_colab.ipynb`.

Recorded artifact: `srq_generalization_m9_repeated_systems_train_only.zip`,
SHA-256
`71e84eeddc24066d250d93328b87561ccad0f0b9cac09a619f66c128b3f7167f`.
All nine gates pass. Across four paired runs, SRQ uses 21.88% of Exact's
persistent state, 20.01% of its serialized-checkpoint bytes, 77.95% of its
analytic PyTorch allocated peak, and 86.09% of its process-attributed NVML
whole-process peak. Its analytic stage is 1.814 times slower, whereas the sum
of measured pipeline stages is 1.013 times slower. These results remain scoped
to one Tesla T4 and one software stack.

Gate: claims and plots expose the remaining quadratic scaling and slower
update rather than hiding them.

### M10 -- optional GACL generalized-stream adapter

Status: PASS on the locked train-only controlled run. GACL is attempted
only after the equal-budget gate. Its inverse-RLS recurrence is tested against
the additive primal Ridge and joint solutions on streams containing both
exposed and unexposed classes. Its original exact/joint weight invariance is
distinguished from the controlled approximation introduced by quantization.
Because the official setting updates by mini-batch, all M10 backends compress
or update after every matched mini-batch rather than reusing FLY's task-level
schedule.

The locked control uses the official random-linear--ReLU structure, width
5,000, gamma 100, five-task fixed Si-Blurry `N=50, M=10` construction and
batch size 64. It uses cached ViT-B/16 features, `online_iter=1`, no image
augmentation, and a train-only development subset, so it is explicitly not an
official end-to-end GACL reproduction. Full definitions and gates are in
`docs/research/SRQ_GENERALIZATION_M10_RUNBOOK.md`; the notebook is
`notebooks/srq_generalization_m10_gacl_colab.ipynb`.

Recorded artifact: `srq_generalization_m10_gacl_train_only.zip`, SHA-256
`8002ac80be9845186b8d9a753d0cf786f0e581b401c927dbf7a0dc95b713208c`.
All ten gates pass. Inverse-RLS and unquantized FP32 square-root have identical
task-boundary accuracy and at least 99.999994% prediction agreement; their
maximum relative weight and logit errors are $3.05\times10^{-5}$ and
$1.00\times10^{-5}$. P2B reduces total state by 72.64% relative to FP32
square-root while losing 0.139 validation-AIA points and 0.200 final-accuracy
points. P2B takes 1.75 times the FP32 square-root update time and 3.79 times
the inverse-RLS reference time over the 35 matched mini-batch updates.

The archived result predates a reporting-only fix: compressed factor tensors
named `factor.*` were omitted from `quadratic_persistent_bytes`, so that
derived field is zero for FP16 and P2B. Total and backend state, the reduction
gate, accuracy, timing, and numerical diagnostics are unaffected. The correct
factor payloads derived from the archived backend totals are 25,015,000 bytes
for FP16 and 13,298,596 bytes for P2B.

### M11 -- budget-locked adaptive precision

Status: PASS on the locked train-only run. M6 supplied the trigger: fixed INT8
narrowly failed the width-20,000 retention gate while FP16 passed. M11 tested
one preregistered label-free rule at widths 10,000 and 20,000. It promoted
strict-upper blocks to FP16 by the largest reduction in factor reconstruction
MSE per added byte, using a fixed 25% allowance between all-INT8 and all-FP16
strict-upper payloads. Its uint8 precision mask and all scale metadata were
counted in persistent state.

Recorded artifact: `srq_generalization_m11_adaptive_precision_train_only.zip`,
SHA-256
`65ce03df4da7041833014628b59aac1167f77bde2d64348b9a8a1e4fe09370a7`.
At widths 10k/20k, adaptive-minus-P2B validation AIA is +0.0792/+0.2599
points, while adaptive-minus-Exact is -0.0085/+0.0041 points. Total-state
reduction relative to Exact is 76.39%/79.92%. Adaptive update time is
1.53/1.51 times the corresponding archived P2B time. All formal M11 gates
pass. Full definitions are in
`docs/research/SRQ_GENERALIZATION_M11_RUNBOOK.md`; the source-locked notebook
is `notebooks/srq_generalization_m11_adaptive_precision_colab.ipynb`.

M11 remains a development-only response to the disclosed M6 failure, not a
replacement for M6. It supports the specific 25% allocation only and does not
establish an optimal policy or resolve GACL mini-batch-frequency drift.

### M11b -- same-byte INT8 scale refinement

Status: formal FAIL; diagnostic complete. This one-shot follow-up kept every
strict-upper value in INT8 and preserved exactly P2B's checkpoint tensor
shapes and dtypes. It replaced max-absolute scaling with four deterministic
alternating least-squares scale/code steps, accepting a candidate group only
when reconstruction error did not increase. No mask, residual, labels,
accuracy, or extra persistent state was used.

M11b is locked to both the M6 and M11 artifacts. At widths 10k/20k it reruns
Exact as a source sentinel and evaluates only the refined all-INT8 method.
The decisive systems gate was byte equality with archived P2B after every
task; the decisive accuracy gates were no worse than P2B and no more than
0.25 point below Exact. Full definitions are in
`docs/research/SRQ_GENERALIZATION_M11B_RUNBOOK.md`; the notebook is
`notebooks/srq_generalization_m11b_scale_refined_colab.ipynb`.

Recorded artifact: `srq_generalization_m11b_scale_refined_train_only.zip`,
SHA-256
`f33153c24716a7c660044cacc1e56cf040ee63d314fd13be2c7d47cf4895a9cf`.
All source, byte-equality, local-error, P2B-improvement, and solver gates pass.
Refinement reduces the same-input local factor error by approximately
1.8--2.0% at every task and improves validation AIA over P2B by 0.0368/0.0057
points at widths 10k/20k. At 10k its Exact-relative loss is 0.050865 points.
At 20k the loss is 0.250115 points, exceeding the locked 0.25-point gate by
0.000115 points; the formal status therefore remains
`FAIL_M11B_SCALE_REFINED_INT8_TRAIN_ONLY`. Refined update time is 1.080/1.046
times P2B and state is byte-identical to P2B.

The threshold is not rounded or relaxed. M11b is useful negative evidence:
lower local reconstruction MSE alone is insufficient to recover the 20k
retention gate. It does not supersede M11, whose adaptive precision remains a
distinct higher-state, higher-accuracy Pareto point. No further scale-rule
search is authorized on this development stream. Packed INT4 and error
feedback remain future alternatives, not prerequisites for finalization.

### M12 -- final locked evaluation

Status: complete. The source-locked artifact contains all 36 units and passes
every non-accuracy completion gate. M12 is restricted to the controlled RanPAC
random-ReLU frontend at widths 10,000 and
20,000. It compares Exact Gram, fixed P2B INT8/FP32, and the single M11
adaptive INT8/FP16 policy over six paired class-order/projection replicates.
M11b is excluded because it failed its train-only development gate.

Authorization is created only after the configuration, current commit,
train-cache content, and exact M6/M11/M11b ZIP identities have been verified
while `test.pt` is absent. The authorized extractor then materializes the
official CIFAR-100 test features. All 36 units are resumable, and their state
bytes are checked after every task: Exact and fixed P2B must match M6 exactly,
whereas adaptive state must remain within the M11-locked all-INT8 floor and
25% factor-budget ceiling under the unchanged accounting identity. Completion
has no accuracy gate, no test-time retry, and no post-test method selection.
The complete frozen contract is in
`docs/research/SRQ_GENERALIZATION_M12_PROTOCOL.md`; the executable notebook is
available for both environments as
`notebooks/srq_generalization_m12_locked_test_colab.ipynb` and
`notebooks/srq_generalization_m12_locked_test_kaggle.ipynb`.

Across six paired test replicates, adaptive SRQ changes AIA relative to Exact
by -0.0052 points at width 10,000 and +0.0046 points at width 20,000 while
reducing state by 76.39% and 79.92%. Fixed P2B reduces state by 79.06% and
82.71% but loses 0.1339 and 0.3264 AIA points. Adaptive is therefore the
higher-state, slower accuracy-retaining point; the small positive 20,000-width
difference is not an accuracy-improvement claim. The final artifact SHA-256 is
`02ada7180e66e780be77c7934b87d268f0d171dfee1c318fd3e9ae7e48846b1e`.
The protocol-recovery disclosure in the M12-specific document remains part of
the result.

### M13 -- equal-budget LoRanPAC challenger (formal FAIL)

M13 adds a source-pinned truncated-SVD LoRanPAC control at widths 10,000 and
20,000 under the exact P2B and adaptive total-state budgets. Rank is derived
from bytes before accuracy is observed. The train-only artifact SHA-256 is
`b7cc3e1993b150d829806ac8062b10a2e31ad9c533ef729ce7a806647496d28c`.
All source, cache, rank, and byte contracts pass, and all four units complete.
LoRanPAC does not dominate its matched SRQ backend on the one development
seed, but the archived status remains `FAIL_M13_LORANPAC_TRAIN_ONLY`: the
task-1 raw basis-orthogonality Frobenius residual and solver residual exceed
the preregistered absolute gates. The accuracy fields are therefore retained
as descriptive evidence only. Full results are recorded in
`docs/research/SRQ_GENERALIZATION_M13_RESULT.md`.

### M13-N -- rank-aware numerical audit (PASS)

M13-N is an accuracy-free, test-free diagnostic of the two M13 failures. Its
artifact SHA-256 is
`726853486664cbf26ec109a061585a1effcd90569ce056108e9ff594f94e031d`.
All gates pass. It reproduces M13's raw task-1 metrics, reports raw and
rank-normalized Frobenius plus spectral orthogonality residuals, and checks a
small FP64 oracle. A diagnostic QR reduces orthogonality error by at least
803.65 times and solver residual by at least 730.74 times. This localizes the
failure to FP32 high-rank basis orthogonality and a non-scale-aware raw gate,
not to a demonstrated algebraic error in the adapter.

The QR path changes the represented truncated system if singular values are
left unchanged and is not substituted into the official LoRanPAC path. M13
therefore remains FAIL. M13-N only authorizes a newly preregistered multi-seed
M14 with scale-aware numerical gates. The audit and recovery disclosure are
recorded in `docs/research/SRQ_GENERALIZATION_M13N_RESULT.md`; the M14 contract
is frozen in `docs/research/SRQ_GENERALIZATION_M14_PLAN.md`.

### M14 -- multi-seed equal-byte LoRanPAC comparison (formal FAIL)

M14 completes all 60 preregistered units across six paired seeds, widths
10,000/20,000, and five methods. Ten of eleven gates pass. The sole failure is
the raw task-1 projected-solver residual: its maximum is
`1.9147568e-3`, above the locked `1e-3` threshold; later-task residuals remain
below `4.392e-6`. No seed is removed and the threshold is not changed. Across
all four equal-byte comparisons, mean validation AIA favors the matched SRQ
backend, but these accuracy results are descriptive because the artifact
status remains `FAIL_M14_LORANPAC_MULTISEED_TRAIN_ONLY`. The artifact SHA-256
is `4beb726bf7f29f8569e5c6de630d90284abdf7ae53928471c27bba178d148b0c`.

### M15 -- LoRanPAC task-1 numerical closure (PASS)

M15 is a train-only, prediction-free audit of the failing seed `4105` and a
sentinel seed `4101` at both widths and budgets. Re-evaluating the official
formula in FP64 on the same stored FP32 basis leaves the residual essentially
unchanged (FP64/FP32 ratio `0.99966--1.00025`). A system-preserving
factorization `U=QT`, with core `T diag(s^2) T^T`, reconstructs the same
truncated system within `7.53e-7` relative error and reduces FP64 core backward
error to at most `5.06e-17`. This closes the failure as an interaction between
small FP32 basis non-orthogonality and a diagonal projected formula that
assumes exact orthogonality, rather than a demonstrated error in the general
Ridge adapter.

M15 passes all gates, but it does not retroactively pass M14, change any M14
prediction, or authorize substituting the diagnostic QR path into the
comparator. Its artifact SHA-256 is
`942cd777674d3e1089ef60b7c1835de70b283b00d948b77c43ab673497a6f9c6`;
the full audit is recorded in
`docs/research/SRQ_GENERALIZATION_M15_RESULT.md`.

### M16 -- fresh Cars/ResNet-50 Phase-2 confirmation (PASS)

M16 is the next out-of-development confirmation. It uses the standard
Stanford Cars split, the published RanPAC Cars schedule (16 initial classes,
then nine increments of 20), the official ResNet-50 ImageNet-1K V2 weights,
and the published random-ReLU width of 10,000. Six paired seeds compare Exact
Gram, locked P2B, and the locked 25% adaptive policy. The test cache may be
materialized only after train-only Ridge selection and an immutable
authorization record.

The experiment is source-pinned to RanPAC commit
`cf4b301d18b0c27db030f4371b72b768005ae58a` and configuration row 10 in
`args/cars_publish.csv`. It is a protocol-faithful Phase-2 backend comparison,
not a full PETL reproduction. RanPAC's code retunes Ridge at every task;
because an SRQ factor embeds `sqrt(lambda) I`, M16 instead applies the official
first-task 80/20 MSE grid once and freezes the selected value for the stream.
Every backend within a replicate receives that same value. There is no test
accuracy gate or post-test retry.

All 18/18 units complete and all locked gates pass. P2B reduces final state by
69.70% while changing paired AIA by `-0.0663 +/- 0.0594` points. Adaptive
reduces state by 67.35% while changing paired AIA by only
`-0.0006 +/- 0.0076` points, at 3.64 times Exact's analytic-update time. The
artifact SHA-256 is
`9d904939c2e3dfcb1ee62d2e39223508833f6855d2eb77aab3566a2dcb6b653a`.
The frozen contract is in `docs/research/SRQ_GENERALIZATION_M16_PROTOCOL.md`;
the audited result is in `docs/research/SRQ_GENERALIZATION_M16_RESULT.md`.

## Naming gate

- FLY evidence only: `SRQ-FLY`.
- FLY plus a passing RanPAC integration: a reusable backend demonstrated on
  two analytic learners.
- FLY, RanPAC, and a passing independent GACL adapter: a generic backend for
  the explicitly defined fixed-feature additive-Ridge family, demonstrated in
  both class-disjoint and generalized streams.

The term `universal plug-in` is prohibited.
