# Phase B SFT-CL implementation record

Status: implementation and synthetic/cache smoke tests only. No CIFAR-100 test
metric is recorded in this document.

## Added components

- `methods/sft_cl`: bounded `G,Q,n` statistics; standard/confusion Fisher
  scatter; hard and full-rank soft transport; analytic global Ridge.
- `methods/cached_replay_baselines.py`: cache-native FLY and current-SOHO
  controls. `CachedSOHOReplay` deliberately persists `feature_history` and
  `label_history`; `metrics.json` labels it `exemplar_free: false`.
- `tools/experiment_runner.py`: explicit cache method dispatch, train-only
  selection grids for `lambda,kappa,delta`, per-task state-byte accounting, and
  optional JSON method config.

## Required Phase B procedure

The frozen cache must be created once with the already verified ViT checkpoint.
It is experiment infrastructure on disk, not learner state.

First select only the proposed SFT configuration from cached **training**
features. This command never opens `test.pt`:

```powershell
python tools/experiment_runner.py --select-config --config configs/sft_cl_cifar100_template.json --feature-cache-dir <CACHE> --output-dir <OUTPUT>/selection --search-methods confusion_fisher_soft --search-lambdas 0.01,0.1,1.0 --search-kappas 0.01,0.1,1.0 --search-deltas 0.01,0.1,0.5 --validation-fraction 0.10 --selection-output <OUTPUT>/selection/soft_confusion.json
```

Lock the resulting `(lambda,kappa,delta)` before opening test features. Run
`fisher_soft` and `shuffled_confusion_fisher_soft` using that exact locked tuple.
For hard Fisher, select only `rank` and `lambda` in a separate train-only call.
The baseline `sft_raw_ridge` uses the locked Ridge policy; legacy FLY/SOHO
reference configurations must be recorded separately rather than tuned on test.

Example final command, after substitution with values from the selection JSON:

```powershell
python tools/experiment_runner.py --config configs/sft_cl_cifar100_template.json --method confusion_fisher_soft --feature-cache-dir <CACHE> --output-dir <OUTPUT>/final_confusion_soft --ridge-lambda <LAMBDA> --fisher-kappa <KAPPA> --fisher-delta <DELTA>
```

Every final run writes `accuracy_matrix.csv`, `task_accuracies.csv`,
`state_bytes.csv`, `timing.csv`, `code_diagnostics.json`, and `metrics.json`.
Report disk cache, runtime memory, and persistent state separately.

## Tests run during implementation

```powershell
python -m pytest -q tests/test_sft_cl_math.py tests/test_experiment_runner.py
python -m pytest -q tests/test_cached_replay_baselines.py
```

Both commands passed at implementation time. Full-suite and CIFAR cached-feature
tests remain required before a Phase B result claim.
