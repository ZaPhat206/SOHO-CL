# SRQ generalization M4 RanPAC train-only runbook

Status: `PASS_M4_RANPAC_TRAIN_ONLY`. The real CIFAR-100 GPU gate passed without
reading or materializing the held-out test split.

## What M4 establishes

M4 asks whether the generic square-root backend works behind a genuinely
different high-dimensional analytic frontend. It uses the official RanPAC
Phase-2 equations, not FLY projection or WTA:

\[
H=\operatorname{ReLU}(FW_{\rm rand}),\qquad
G\leftarrow G+H^\top H,\qquad
Q\leftarrow Q+H^\top Y.
\]

The upstream implementation is locked to
[`cf4b301d`](https://github.com/McDonnell-Research-Lab/RanPAC/tree/cf4b301d18b0c27db030f4371b72b768005ae58a),
whose `RanPAC.py` SHA-256 is
`36ba66a6ce6993b5830539a511e84dbd2b4994d8ebe2a0d5024ec5438a427e11`.
The associated method is described in the
[official NeurIPS 2023 paper](https://papers.neurips.cc/paper_files/paper/2023/file/2793dc35e14003dd367684d93d236847-Paper-Conference.pdf).

This is a controlled analytic-head study. It does not reproduce RanPAC PETL.
The original code selects Ridge again from the current task at every update;
M4 instead selects one fixed value on a disjoint calibration subset of the
training partition. A fixed value is necessary for all backends to represent
the same additive system throughout the comparison. Therefore a passing M4
supports backend portability, not reproduction of the published RanPAC
accuracy.

## Locked comparison

All paths receive byte-identical expanded features, labels, class order,
task grouping, projection, Ridge coefficient, and validation indices:

1. independent direct implementation of the original Gram/Ridge equations;
2. generic Exact Gram backend;
3. dense FP32 square-root backend with blocked QR;
4. FP16 square-root storage;
5. frozen P2B mixed INT8/FP32 square-root storage.

The random projection has shape `768 x 10000`, standard-normal entries, seed
2025, and a ReLU activation. The outer development split is deterministic and
stratified, with 80% used for updates and 20% used only for validation. The
Ridge calibration subset is drawn only from the outer update partition: 20
fit and 5 validation samples per class. The full upstream candidate grid
`10^-8,...,10^8` is scored by mean squared error. The selected value is then
frozen before any outer validation accuracy is computed.

## Gates

- direct reference and generic Exact tensors and persistent bytes are
  identical after every task;
- FP32 square-root has system error at most `5e-5`, weight/logit error at most
  `1e-3`, and prediction agreement at least `0.999` against Exact;
- P2B reduces quadratic persistent state by at least 70%;
- P2B validation-AIA loss relative to generic Exact is at most 0.20 percentage
  points;
- every solver residual is at most `1e-5` and no numerical failure occurs.

The gate is source-locked. A failure cannot be repaired by inspecting test
accuracy, changing the selected Ridge after outer evaluation, or relaxing a
tolerance.

## Run on Colab

Open and run every cell in order:

`notebooks/srq_generalization_m4_ranpac_colab.ipynb`

The notebook performs a fresh clone, verifies source SHA-256 values, runs the
synthetic suite, creates only `train.pt`, runs the ten-task gate, and exports:

`srq_generalization_m4_ranpac_train_only.zip`

The archive must contain only `m4_results.json` and `config.json`. Do not add
the feature cache, projected codes, model checkpoint, or any test tensor.

## Interpretation

A PASS means SRQ has been demonstrated on two distinct analytic frontends:
FLY and RanPAC Phase-2. The allowed claim becomes “a reusable backend
demonstrated on two analytic learners.” It is still not a universal plug-in.

A FAIL is also informative: the paper remains scoped to SRQ-FLY. The failure
record identifies whether the obstacle is semantics, FP32 square-root
equivalence, quantization accuracy, state reduction, or numerical stability.
No held-out test run is authorized by M4 itself.

## Recorded result

Artifact SHA-256:
`228da0828c7f6964bcc8f7da92258efa20b00eef0fc83679246ce0ae2a0c8f52`.
The result JSON SHA-256 is
`1afb59eeb6047c1dcdde2da14da27c0c8763b1b06bf43baa8f9b38019e39cfcc`.
The archive contains exactly `m4_results.json` and the byte-identical locked
`config.json`. Provenance records clean source commit
`47a789b6656e88d2ae5a3211f34eea485636e10a`.

The train-only calibration selected `lambda=1e6` from the complete locked
grid. Exact and dense FP32 square-root both reached validation AIA `92.6222`
and final validation accuracy `88.50`. FP16 square-root reached AIA `92.6215`
and final `88.53`. P2B reached AIA `92.4608` and final `88.39`, corresponding
to an Exact-minus-P2B AIA gap of `0.1614` percentage points.

The final Exact/P2B total persistent states are `438720400 / 91880088` bytes,
a total-state reduction of `79.0573%`. The corresponding quadratic states are
`400000000 / 53159688` bytes, a reduction of `86.7101%`. The dense FP32
square-root preserves 100% prediction agreement with Exact; its maximum
system, weight, and logit relative errors are `2.10e-6`, `1.24e-5`, and
`9.43e-5`. The maximum solver residual over every path and task is `4.65e-6`.
All eight preregistered gates passed and `numerical_failure` is null.

This result changes the defensible scope: SRQ is now a reusable additive-Ridge
backend demonstrated behind both FLY/WTA and RanPAC/random-ReLU frontends. It
does not justify the phrases "universal plug-in" or "full RanPAC reproduction."
