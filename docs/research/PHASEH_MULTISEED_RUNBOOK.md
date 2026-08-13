# Phase H-B — locked multi-seed Colab runbook

Status: runner/notebook implementation gate **PASS**. The five-seed CIFAR-100
study has not been executed.

Protocol amendment 1 was registered before observing any Phase H result. The
paper value `93.89` is now an external reproduction diagnostic, not an
acceptance gate for the matched internal study. A single seed from the current
source/runtime is not statistically equivalent to the paper's reported mean,
and checkpoint/environment/source differences can affect absolute accuracy.
The runner always records the discrepancy and continues all locked methods.
If it exceeds `0.5` points, the run must not be called a paper reproduction;
it remains usable as a shared-cache, paired internal comparison. No method,
seed, rank, Ridge value, feature, or evaluation metric changed in this
amendment.

Use `notebooks/phaseh_multiseed_cifar100_colab.ipynb` from branch
`feature/crt-soho`. Before running, place the exact returned Phase G artifact
at `MyDrive/T-SOHO/schur_locked_heldout_results.zip`. Its required SHA-256 is
`9ecaa259deb998f36abdd8052145b17a0ce84adeeb2168b29a83c039868cbc77`.

Run all eight cells in order. Edit paths only in Cell 2. The notebook:

1. mounts Drive and clones the selected branch;
2. verifies Phase G evidence before opening the feature cache;
3. restores the shared cache from Drive, or extracts and saves it once;
4. runs the focused correctness suite;
5. runs all methods in their declared normal order;
6. records whether FLY AA differs from `93.89` by more than `0.5` points;
7. completes the locked five-seed/eight-method grid regardless of that
   external-reference discrepancy;
8. reports mean/std and paired 95% t intervals and downloads a ZIP.

Progress is deliberately compact:

```text
[start | seed 2/5=2025 | method 4/8=schur_residual] device=cuda
[seed 2/5 | method 4/8=schur_residual | task 7/10] stage=UPDATE
[seed 2/5 | method 4/8=schur_residual | task 7/10] stage=EVAL seen_tasks=7
[seed 2/5 | method 4/8 | task 7/10] elapsed=00:01:42 unit_eta=00:00:43 study_eta=01:18:20
```

An `UPDATE` line may remain visible while a large analytic Gram/Cholesky solve
is running; it is not by itself evidence that Colab has frozen.

One JSON is atomically stored on Drive after each complete seed/method unit.
After a disconnect, rerun Cells 3–7; validated complete units are skipped. An
interrupted in-progress method restarts from its first task because learner
checkpoints for FLY/SOHO would be large and SOHO checkpoints deliberately
contain sample-level replay.

Resume validation rejects a result whose manifest/cache identity, seed,
method, class order, exemplar disclosure, task matrix shape, or numerical
metrics are incomplete or inconsistent. The output also stores byte-identical
copies of the locked Phase H manifest and Phase G evidence ZIP. The runner
cannot accept rank, Ridge, seed, or method-search arguments.

The feature cache is experiment infrastructure. FLY and Schur are
exemplar-free in learner state; the matched current-SOHO control is not and
reports all replay bytes. Do not use the displayed test results to change
seeds, ranks, Ridge ranges, anchor parameters, or preprocessing. Do not proceed
to another dataset until the returned ZIP is audited.

Runtime columns measure only classifier learner update/inference inside this
cached-feature runner. `peak_runtime_memory_bytes` is PyTorch CUDA allocated
memory for the active method; it is not process RSS, feature-cache disk size,
or a paper-runtime comparison. The notebook uses the same cache for every
method, while that cache remains experiment infrastructure rather than learner
state.

## Implementation verification

Exact focused command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_phaseh_multiseed.py tests/test_cached_replay_baselines.py tests/test_experiment_runner.py
```

Result after protocol amendment: `24 passed`, exit code `0`, pytest runtime
`5.56s`.

Exact full-suite command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result after protocol amendment: `82 passed`, exit code `0`, pytest runtime
`8.31s`.

The warnings were 18 PyTorch JIT deprecations, one sparse-CSC beta warning,
and one sparse-invariant warning. They were not suppressed and no test failed.
The exact supplied Phase G ZIP was also checked locally: its ZIP SHA-256,
inner gate SHA-256, and selected Schur/raw configurations all passed the
locked authorization checks. These are correctness/integrity checks only;
they do not substitute for the pending five-seed CIFAR-100 run.
