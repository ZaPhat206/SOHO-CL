# Phase C — CRT-SOHO mathematical prototype

Status: mathematical and integration gate **PASS**; empirical CIFAR-100 gates
**NOT RUN**. This phase does not establish an accuracy improvement and does
not authorize held-out test evaluation.

## Implemented scope

The new learner is selected only through explicit `crt_*` method names. No
existing SOHO, FLY, T-SOHO, or SFT learner implementation was changed.

Implemented controls:

- `crt_anchor_only`;
- `crt_full_raw_residual`;
- `crt_random_residual`;
- `crt_fisher_residual`;
- `crt_confusion_residual` (the proposal);
- `crt_shuffled_confusion_residual`;
- `crt_confusion_no_residualization`.

The persistent sufficient statistics are `G_pp:(M,M)`, `G_xx:(D,D)`,
`H_px:(M,D)`, `Q_p:(M,C_seen)`, `Q_x:(D,C_seen)`, and
`counts:(C_seen,)`. The fixed sparse anchor projection is `(M,D)`. Current
derived classifiers/directions are bounded by `D`, `M`, `C_seen`, and `r`.
No tensor has a historical sample-count dimension. Checkpoints contain only
configuration, fixed anchor projection, class mapping/counts, and sufficient
statistics; derived tensors are reconstructed after loading.

All Ridge/complement/block systems use Cholesky solves. No explicit matrix
inverse was added. Inference is one global seen-class call and the
`predict_logits` signature has no `task_id`.

## Exact test commands and results

Environment: Windows, Python 3.13, PyTorch from the active repository
environment. Synthetic tests use CPU and `torch.float64` unless a test is
explicitly checking the fixed FLY anchor, whose projection/forward path is
`float32` before accumulation into `float64` statistics.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/test_crt_soho_math.py tests/test_experiment_runner.py -q
```

Result: `23 passed`, exit code `0`, runtime `5.56s`. The only method-related
warning is PyTorch's existing sparse-CSC beta warning from `FlyHash`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `52 passed`, exit code `0`, runtime `6.61s`. There were 19 warnings:
18 existing `torch.jit` deprecation warnings and one existing sparse-CSC beta
warning.

## Numerical checks

Synthetic generators use fixed seeds `2, 4, 6, 8, 9, 17, 23, 31, 41`.
Tolerances are:

- streaming versus batch dual-view statistics: `atol=rtol=1e-12`;
- reconstructed versus explicit historical residual statistics:
  `atol=rtol=1e-11`;
- Schur/block solution versus a direct augmented-design solve:
  `atol=rtol=1e-10`, with maximum equation residual below `1e-10`;
- checkpoint/rebuild logits: `atol=rtol=1e-10`.

The test suite also establishes that:

1. class-column expansion is canonical and streaming statistics equal their
   batch oracle;
2. the reconstructed `G_pr`, `G_rr`, and `Q_r` equal statistics computed from
   an explicitly materialized historical residual matrix;
3. the block Schur solver equals direct augmented Ridge with separate anchor
   and residual penalties;
4. confusion affinity is symmetric/non-negative with zero diagonal, and its
   shuffled control preserves the undirected edge-value multiset;
5. every control returns finite global logits, has no Task-ID inference
   argument, passes the state audit, and round-trips through its checkpoint;
6. on a deliberately constructed toy problem, an anchor that discards the
   label-bearing raw coordinate obtains `50%`, while the full-raw residual
   obtains `100%`. This is a feasibility witness only, not empirical evidence
   for CIFAR-100.

## Conditions and limitations

- Exact residual reconstruction requires both views and the anchor map to be
  fixed over history. Changing the backbone, random projection, WTA rule, or
  preprocessing invalidates the retained cross-statistics.
- The reconstruction theorem is algebraic; it does not imply that a residual
  improves generalization or forgetting.
- Fisher directions are recomputed from all retained class-level statistics.
  Their rank is capped by `min(requested_rank, D, C_seen-1)` except for the
  explicit full-raw control.
- Euclidean QR normalizes Fisher directions so that direction scaling cannot
  silently act as an extra Ridge hyperparameter. Only the selected subspace is
  compared.
- Confusion is estimated from class-mean expected margins of the analytic
  anchor classifier. This is a class-level proxy, not the full per-sample
  confusion distribution.
- The template values in `configs/crt_soho_cifar100_template.json` are
  provisional search starting points, not selected or reported parameters.
- The current generic grid runner is correctness-oriented and repeats anchor
  encoding/statistic accumulation per candidate. A large grid would be
  wasteful. The next implementation step is a reusable on-disk anchor/statistic
  cache for train-only selection; that cache remains experiment infrastructure
  and must never enter learner checkpoints.

## Gate decision and next permitted work

Mathematical prototype and regression integration: **PASS**.

Next permitted work is only an efficient train-only CIFAR-100 gate runner:

1. select anchor Ridge for `crt_anchor_only`;
2. lock it and test whether `crt_full_raw_residual` improves validation AA;
3. only if gate 1 passes, compare the low-rank proposal with random,
   standard-Fisher, shuffled-confusion, and no-residualization controls using
   the same locked split and budgets;
4. do not open cached test features unless all predeclared gates pass.

No CIFAR-100 test accuracy, multi-seed result, or paper-level improvement is
claimed in this phase.
