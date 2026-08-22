# SRQ-FLY manuscript validation

Validation date: 2026-08-21. No dataset, feature cache, WTA cache, or held-out
example was opened by these checks.

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
