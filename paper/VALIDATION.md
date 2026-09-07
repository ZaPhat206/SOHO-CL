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
