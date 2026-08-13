# Phase D — optimized CRT-SOHO train-only gate runner

Status: runner correctness gate **PASS**; CIFAR-100 validation gates **NOT
RUN**; held-out test remains unauthorized.

## Outcome

`tools/crt_gate_runner.py` now separates expensive fixed-view construction
from analytic candidate evaluation:

1. it opens only the frozen-feature `train.pt` cache;
2. it computes the fixed sparse/WTA anchor view once;
3. it saves one cumulative sufficient-statistic snapshot per task and only
   the anchor features/indices of the deterministic validation subset;
4. it hashes every experiment-cache artifact and writes `metadata.json` last
   as the completion marker;
5. it restores snapshots and solves candidates without re-encoding historical
   training features;
6. it stops immediately after a failed Gate 1 and never launches held-out
   test evaluation.

The new public cached-view learner calls do not retain their inputs:
`encode_anchor`, `update_from_views`, `predict_logits_from_views`, and
`restore_sufficient_statistics`. Checkpoint contents and exemplar-free state
invariants are unchanged.

## Staged selection policy

- Select anchor Ridge for `anchor_only`.
- Lock anchor Ridge; select residual/complement Ridge for
  `full_raw_residual`.
- If full residual does not clear the predeclared gain, stop.
- Lock all three Ridge values; search rank/temperature for
  `confusion_residual`.
- Search every control over the same applicable train-validation budget:
  random and standard Fisher use the rank grid; shuffled-confusion and
  no-residualization use the rank/temperature grid.
- Gate 3 compares the proposal to the strongest selected control, making the
  falsification test conservative toward CRT-SOHO.

Default accuracy thresholds are expressed in percentage points: full-residual gain
`>=0.10`, low-rank gap from full residual `<=0.50`, and confusion gain over the
strongest control `>=0.10`. They are recorded in `gate_results.json` and must
not be changed after viewing held-out results. Numerical Gate 0 additionally
requires maximum relative solver residual `<=1e-4` for every evaluated
candidate.

## Exact commands and results

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/test_crt_gate_runner.py tests/test_crt_soho_math.py tests/test_experiment_runner.py -q
```

Result: `27 passed`, exit code `0`, runtime `5.91s`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `56 passed`, exit code `0`, runtime `6.39s`.

Warnings: 18 existing `torch.jit` deprecations, the existing PyTorch
sparse-CSC beta warning, and one sparse-invariant warning raised while loading
the synthetic CSC cache. No test failed.

The synthetic gate test physically renames `test.pt` before cache construction
and evaluation. All stages still complete and every candidate records
`uses_test_set=false`. Separate tests establish fail-closed SHA-256 integrity,
Gate-1 early stopping, JSON result completeness, and full learner/checkpoint
math regression.

## Colab artifact

`notebooks/crt_soho_cifar100_colab.ipynb` contains nine cells and was parsed as
valid notebook schema version 4. It performs checkpoint preflight, reuses or
extracts the frozen ViT cache, runs only train-validation gates with visible
progress, displays candidates, and downloads result evidence. It contains no
held-out evaluation cell.

The default pilot uses anchor dimension 1024 and float32 statistics for T4
feasibility. These are pilot settings, not paper claims. Solver residuals are
logged for every candidate. An unacceptable numerical residual fails the
scientific interpretation even if accuracy gates appear to pass.

## Storage classification

The CRT gate cache may contain sample-indexed validation features and indices.
It is therefore explicitly experiment infrastructure and is forbidden from
learner checkpoints. It is not evidence that the learner itself uses replay.
The learner remains exemplar-free only because its checkpoint inventory is
limited to the fixed projection, sufficient statistics, class mapping/counts,
configuration, and bounded derived state.

## Gate decision

Implementation/test gate: **PASS**.

Next action: push the branch, run the Colab notebook once, and return
`gate_results.json`. Regardless of its value of `held_out_test_authorized`, stop
for review before any held-out test run. No improvement claim is currently
supported.
