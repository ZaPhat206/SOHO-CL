# SRQ-FLY manuscript validation

Initial validation date: 2026-08-21. Priority-5 artifact audit date:
2026-09-06. No dataset, feature cache, WTA cache, or held-out example was
opened by these checks.

## SRQ generalization M0 baseline lock

On 2026-09-07, the current paper evidence was frozen in
`paper/EXPERIMENTS_MANIFEST.json` before generic-backend refactoring. The
manifest gate passed `4/4`, the focused SRQ regression suite passed `77/77`,
and the full repository suite passed `478/478`. Exact commands, environment,
worktree caveat, and warning classes are recorded in
`docs/research/SRQ_GENERALIZATION_M0_BASELINE_LOCK.md`. M0 authorizes local M1
refactoring only; it does not authorize a held-out or dirty-worktree run.

## SRQ generalization M1 generic backend extraction

On 2026-09-07, focused generic-backend and legacy regression tests passed
`59/59`; the complete repository suite passed `487/487`. The extracted
backend is representation-agnostic, reproduces version-1 P2B compressed state
for FP16 and INT8, imports legacy checkpoints, and preserves total FLY learner
bytes through a frontend wrapper. Exact commands and scope are recorded in
`docs/research/SRQ_GENERALIZATION_M1_BACKEND_EXTRACTION.md`.

## Priority-5 whole-process memory artifact

The returned artifact `srq_fly_priority5_whole_process_memory.zip` was audited
read-only. Its SHA-256 is
`7f111e80ec3e4d12fafae39a868795fc36c967d223c99f8ade98107b5b180403`.
ZIP CRC validation passed; the SHA-256 of `priority5_memory_results.json`
matches the internal manifest; the locked config hash is
`ba02e0e742fdaf17e364d6ef182d8e32f8220ac5140253ec2155572054db75da`;
and the source commit is
`a3dbe581e2b7c61d54203139c5d07649ec7dbfd5`.

The result status is `PASS_PRIORITY5_MEMORY`; all ten preregistered gates are
true. Both isolated workers completed, remained train-only, used paired data
and projection identities, and were observed by NVML across all required
stages. The audit records a 0.21884 state ratio, 0.77953 analytic PyTorch peak
ratio, 0.86156 analytic NVML worker-peak ratio, 0.86090 whole-process NVML
worker-peak ratio, and 0.99609 fixed-probe prediction agreement. This validates
the reported CIFAR-100/Tesla-T4 systems result only; it does not convert the
train-only probe into an accuracy result.

## Focused SRQ regression suite

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_fly_math.py tests/test_srq_fly_learner.py tests/test_srq_fly_d2_state_match.py tests/test_srq_fly_d21_lambda_robustness.py tests/test_srq_fly_d3_cub.py tests/test_srq_fly_d4_cub_multiseed.py
```

Result: `40 passed, 20 warnings in 34.49s`. Warnings were PyTorch JIT
deprecations and existing sparse CSC/invariant warnings; no test failed.

## Full repository suite

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Initial manuscript result: `253 passed, 20 warnings in 56.19s`.

After adding the feature-free ImageNet-R identity audit and its six synthetic
tests, the exact command was rerun. Latest result:
`259 passed, 20 warnings in 31.24s`.

After adding the CIFAR D5 train-only selection runner and four synthetic
tests, the exact command was rerun. Latest result:
`263 passed, 20 warnings in 58.14s`.

After adding fail-closed handling for a numerically invalid fixed-Ridge inner
candidate, two more synthetic tests were added. Latest result:
`265 passed, 20 warnings in 28.40s`.

After adding the immutable three-dataset held-out manifest, single-use
authorization, test-only extractor, multi-seed evaluator and notebook checks,
the exact full-suite command was rerun. Latest result:
`277 passed, 20 warnings in 21.65s`.

## Three-dataset held-out runner tests

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_fly_heldout.py
```

Result: `12 passed, 19 warnings in 5.31s`. Tests cover manifest/source
identity, real train-only evidence semantics, refusal of test-contaminated
selection evidence, idempotent authorization, metric definitions, persistent
sample-state rejection, realized sparse-state accounting, paired SRQ/exact
evaluation, state-matched and raw controls, test-cache authorization,
aggregation without an accuracy gate, and compilation/hash binding of every
Colab code cell.

The real train-only ZIPs were verified read-only with:

```powershell
python -u tools/srq_fly_heldout.py verify-selection `
  --manifest configs/srq_fly_three_dataset_heldout.json `
  --artifact-dir C:\Users\Admin\Downloads
```

All four artifacts matched their locked sizes, SHA-256 identities, statuses,
`uses_test_set=false` contracts and selected Ridge values. No held-out feature
or dataset was opened by local validation.

## CIFAR D5 train-only gate tests

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_fly_cifar_selection.py tests/test_srq_fly_math.py tests/test_srq_fly_learner.py
```

Initial result: `20 passed, 20 warnings in 21.44s`. After the failed-candidate
contract tests were added, the result was
`22 passed, 20 warnings in 8.38s`. The warnings are the existing
PyTorch JIT deprecations and sparse CSC/invariant notices. The synthetic
end-to-end test verifies selection, paired exact/SRQ evaluation, state
accounting, resume behavior, and refusal of a visible `test.pt`.

## ImageNet-R dataset-audit tests

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_imagenetr_dataset_audit.py tests/test_cub_dataset_audit.py tests/test_data_utils.py tests/test_cub_data_utils.py
```

Result: `16 passed in 12.47s`.

The real raw-byte/path audit used the exact command recorded in
`docs/research/SRQ_FLY_IMAGENETR_DATASET_AUDIT.md`. It returned the intentional
exit code `2` with `FAIL_CROSS_SPLIT_DUPLICATES`; zero images were decoded and
zero features were extracted.

## Document consistency checks

The review checked:

- balanced Markdown LaTeX delimiters in the manuscript files;
- every `[@key]` citation exists in `references.bib`;
- balanced BibTeX braces;
- `git diff --check`;
- exact nominal state projection using
  `projected_srq_state_bytes(feature_dim=768, expand_dim=10000,
  synaptic_degree=300, num_classes=200, block_size=256, group_size=64)`.

The nominal projection gives `105166640` SRQ bytes and `452006952` exact-FLY
bytes. The audited runtime values are 12 bytes lower because the realized CSC
projection stores 2,999,999 rather than the nominal 3,000,000 nonzeros. The
proof appendix therefore uses the actual stored nonzero count \(\nu\), not an
unqualified \(ms\), in its exact byte equation.

## Self-contained final-notebook tests

The notebook that performs train-only nested selection before the first
three-dataset test run was checked with:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_fly_selfcontained.py tests/test_srq_fly_heldout.py tests/test_srq_fly_math.py tests/test_srq_fly_learner.py
```

Result: `36 passed, 19 warnings in 8.45s`. This gate checks the immutable
protocol and source hashes, disjoint nested partitions, refusal of a visible
`test.pt`, train-only selection, boundary-lambda stopping, authorization
binding, output aggregation, and compilation of every notebook code cell.

The exact full-suite command was then rerun:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `285 passed, 20 warnings in 24.88s`. The warnings are the existing
PyTorch JIT deprecations and sparse CSC/invariant notices; no test failed.

## Generic analytic Ridge M2 equivalence gate

The source-locked M2 audit compares Exact Gram, dense QR, and blocked QR on
Gaussian, column-scaled, and sparse synthetic streams in FP64 and FP32. It
uses no dataset or test split. Full details and per-case maxima are recorded in
`docs/research/SRQ_GENERALIZATION_M2_EQUIVALENCE.md`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_analytic_ridge_equivalence.py tests/test_analytic_ridge_backend.py
python -u tools/srq_generalization_m2.py run `
  --config configs/srq_generalization_m2_equivalence.json `
  --output tmp/srq_generalization_m2_results.json
```

Result: `12 passed, 1 warning in 6.45s` and
`PASS_M2_UNQUANTIZED_EQUIVALENCE`. Prediction agreement was 100% in all six
case/precision combinations; the maximum FP32 reconstructed-system error was
`1.95e-7`. The subsequent full repository run completed with
`490 passed, 20 warnings in 105.22s`.

## Generic analytic Ridge M3 local preflight

The real CIFAR train-only regression is prepared but not yet reported as a
PASS. Its source lock, protocol, expected artifact, and interpretation are in
`docs/research/SRQ_GENERALIZATION_M3_RUNBOOK.md`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_generalization_m3.py `
  tests/test_analytic_ridge_backend.py `
  tests/test_analytic_ridge_equivalence.py
```

Result: `16 passed, 19 warnings in 10.95s`. The subsequent full repository
run completed with `494 passed, 20 warnings in 121.72s`. This authorizes the
source-locked Colab preflight, not M4 and not any held-out evaluation.

The returned M3 artifact was then audited read-only. ZIP SHA-256 is
`ec376875ff0e29e2d7a46d1fcb9fcccc523ea47fbe6a8f2dd83ec3a98930a595`;
it contains only the result and byte-identical locked config. All ten records,
all five gates, source hashes, clean commit identity, train-only contract, and
state accounting passed. Final decision: `PASS_M3_FLY_REGRESSION`.

## Generic analytic Ridge M4 RanPAC gate

The source-locked M4 implementation was checked locally before the GPU run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_generalization_m4.py `
  tests/test_ranpac_analytic_frontend.py `
  tests/test_analytic_ridge_backend.py `
  tests/test_analytic_ridge_equivalence.py
python -m pytest -q
```

Results: `23 passed, 19 warnings` for the focused suite and `505 passed, 20
warnings` for the full repository suite. The returned artifact was then read
and decompressed without extracting sample data. ZIP SHA-256 is
`228da0828c7f6964bcc8f7da92258efa20b00eef0fc83679246ce0ae2a0c8f52`;
it contains exactly `m4_results.json` and `config.json`. The config is
byte-identical to the locked repository file. Result SHA-256 is
`1afb59eeb6047c1dcdde2da14da27c0c8763b1b06bf43baa8f9b38019e39cfcc`.

The artifact reports clean source commit `47a789b`, ten task records,
`uses_test_set=false`, null numerical failure, and all eight gates true. Source
and split identities match the locked protocol. Final decision:
`PASS_M4_RANPAC_TRAIN_ONLY`.

## Generic analytic Ridge M5 equal-budget gate

The source-locked M5 implementation was checked locally before the GPU run:

```powershell
$env:PYTHONDWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_generalization_m5.py `
  tests/test_countsketch_analytic_frontend.py `
  tests/test_srq_generalization_m4.py `
  tests/test_ranpac_analytic_frontend.py `
  tests/test_analytic_ridge_backend.py `
  tests/test_analytic_ridge_equivalence.py
python -m pytest -q
```

The focused preflight passed, and the repository full suite before the run was
`516 passed, 20 warnings`. A cross-platform source-lock defect was then found
before the experiment began: Windows CRLF bytes had been recorded for
`tools/experiment_runner.py`, while Colab checked out canonical LF bytes. The
lock was corrected without changing the runner or scientific protocol, and a
regression test now hashes universal-newline text. The focused post-fix suite
reported `11 passed, 18 warnings`.

The returned ZIP was audited read-only. SHA-256 is
`0f5e23fa4a7d83926638641025fb103895561073cd1f0fa4e1e0d552fd2fa931`;
it contains exactly `m5_results.json` and `config.json`, passes ZIP CRC, and the
config is byte-identical to the locked repository file. Result SHA-256 is
`9d58e9e27270bfb34ff47276dfa1256231ec2e8a2e619f6c6905968c73c7a541`.

The artifact reports clean source commit `cc7a851`, ten records per method,
`uses_test_set=false`, dimensions locked before accuracy, null numerical
failure, and all seven gates true. The maximum solver relative residual is
`4.65e-6`. Final decision: `PASS_M5_EQUAL_BUDGET_TRAIN_ONLY`.

## Generic analytic Ridge M6 width-scaling gate

The returned ZIP was audited read-only. SHA-256 is
`b2739b9da023ebd2eedb6fdfe01c394e94f252773e847533b35350021c3d239e`;
it passes ZIP CRC and contains exactly `config.json`, `m6_results.json`,
`width_sweep.csv`, `width_sweep_accuracy_state.svg`, and
`width_sweep_update_time.svg`. Both SVG files parse as valid XML. The config is
byte-identical to the locked repository file, the CSV has 21 method-width
rows matching the JSON values, and result SHA-256 is
`648630c5f0b70ed85e675942a8c2fb10e6f33eb3ff8f03e8a71421de22f44cbb`.

The artifact reports clean source commit `d96714e`, all seven widths complete,
`uses_test_set=false`, nested projection prefixes, and no numerical failure.
Six gates pass. The sole failure is `p2b_accuracy_retention`: the maximum loss
is `0.255845` validation-AIA points at width 20k versus the locked `0.25`
limit. The maximum FP16 loss is `0.0100` points and maximum solver residual is
`5.83e-6`. Final decision: `FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY`; retain the result
and investigate it in M7 without relaxing the gate or inspecting test data.

## Generic analytic Ridge M7 task-wise error diagnostic

The returned ZIP was audited read-only. SHA-256 is
`df92adadce046c53efa5b9fcf01435d1fab2a4c71a690b10c3d7c221205acf36`;
it passes ZIP CRC and contains exactly `config.json`, `m7_results.json`,
`error_trajectory.csv`, and `m7_error_trajectory.svg`. The SVG parses as valid
XML, the CSV has 60 method-width-task rows whose scalar values match the JSON,
and all recorded task accuracies exactly reproduce M6 at widths 10k and 20k.
The embedded config is byte-identical to the repository config. Result
SHA-256 is
`1799ad863ab389f67b13ffdb59df55b7dc436d217e5e54c45ad28e6d4eaa0bf8`.

The artifact reports clean source commit `dd92292`, `uses_test_set=false`, and
all five gates true. Maximum Exact probe-identity error is `1.98e-6`; maximum
solver relative residual is `5.83e-6`. At the final task, P2B system-action,
weight, and logit errors are `0.00859 / 0.0272 / 0.3178` at width 10k and
`0.00902 / 0.0488 / 0.6352` at width 20k. Prediction agreement is
`98.46% / 97.24%`, and the sufficient margin-certificate fraction is
`84.19% / 75.43%`. Final decision:
`PASS_M7_ERROR_TRAJECTORY_TRAIN_ONLY` as a diagnostic integrity result. It
does not alter M6's formal failure and does not establish a condition number
or a causal effect.
