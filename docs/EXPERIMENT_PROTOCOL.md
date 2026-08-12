# T-SOHO experiment protocol

Status: protocol for post-implementation work. Do not begin a later phase or a research-scale run until its preceding gate passes and the commands/results are recorded.

## Non-negotiable comparison contract

Every row comparing FLY-CL, current SOHO, raw-feature Ridge, and T-SOHO must use the same:

- frozen backbone and pretrained checkpoint;
- input resize/crop/normalization;
- dataset version and train/test partition;
- global class order and task partition;
- seed, determinism settings, batch size, hardware/device policy where timing is reported;
- evaluation schedule and global seen-class inference rule.

Store a machine-readable run manifest with exact CLI/config, git commit, dirty-worktree status, Python/PyTorch/timm versions, backbone identifier/checkpoint hash, dataset location/version, class-order list/hash, seed, and metric outputs. A method-specific hyperparameter must be selected using only train-time information or a declared validation split. Test accuracy must never choose `r`, `τ`, `λ`, preprocessing, or a stopping point.

Current repository note: `main.py` does not implement `--config`, and `configs/flycl_cifar100.yaml` is empty. Before experiments, Phase 1 must add/configure a real explicit config path; do not describe a YAML command as executable before that work lands.

## Metrics and memory accounting

Report AA, final-stage `A_T`, learning accuracy, forgetting, BWT, per-task accuracy matrix, mean/total train time, feature-extraction time, and peak CUDA allocation. Separately report:

1. learner persistent-state bytes excluding frozen backbone;
2. sample-level retained bytes (must be exactly zero for T-SOHO);
3. temporary peak solver/workspace bytes, if measurable;
4. checkpoint size and a checkpoint-state inventory.

Do not infer exemplar-free status from low GPU memory alone.

## Phases, gates, and exact command templates

The commands below become mandatory once their referenced files exist. Replace only documented placeholders; record stdout/stderr and exit status in the phase result. These are deliberately small tests, not large experiments.

| Phase | Work | Required commands | Gate to proceed |
|---|---|---|---|
| 0 — baseline lock | Capture current FLY/SOHO commands and task order; add no method. | `git status --short`; `git rev-parse HEAD`; `cd ../LAB_FLY/scripts && ./test_cifar.sh` (only when a baseline reproduction run is approved). | Manifest records clean/dirty status, exact command and all comparison-contract fields. |
| 1 — config + math | Add real T-SOHO config dispatch and pure math functions. | `python -m pytest -q tests/test_tsoho_math.py` | Streaming/statistical and code-geometry unit tests pass; config resolves without modifying SOHO/FLY defaults. |
| 2 — algebra controls | Add raw-Ridge/transport identity controls. | `python -m pytest -q tests/test_tsoho_math.py -k "ridge or transport or etf or orthogonal"` | `P=W_REᵀ`, decoder equation, full-orthogonal control and simplex argmax-equivalence tests pass. |
| 3 — learner state | Add streaming learner and checkpoint inventory. | `python -m pytest -q tests/test_tsohocl_streaming.py`; `python -m pytest -q tests/test_tsoho_checkpoint.py` | No sample-level state/cache/checkpoint field; sequential statistics equal reference; no Task-ID path in inference. |
| 4 — regression integration | Wire CLI and ensure baselines remain compatible. | `python -m pytest -q tests/test_regression_baselines.py tests/test_tsohocl_integration.py` | Existing SOHO/Fly smoke fixtures preserve expected behavior; T-SOHO completes ≥2 toy tasks with global outputs. |
| 5 — approved pilot | Run a small fixed, train-only-selected configuration across all controls. | `python main.py --config configs/t_soho/<pilot>.yaml`; equivalent explicit FlyCL/SOHO/raw-Ridge commands recorded in manifest. | All controls share the contract; checkpoint audit passes; full-rank and random-code controls are reported. |
| 6 — research-scale evaluation | Seeds/datasets/ablations pre-registered. | Exact commands generated from committed configs, e.g. `python tools/run_protocol.py --manifest configs/t_soho/<study>.yaml --seeds 1993 2023 ...` (only if/when that tool is implemented). | Only permitted after Phase 5 report is reviewed and the low-rank/geometry hypothesis survives controls. |

Until tests/config tooling are implemented, the Phase 1–6 commands are specifications, not runnable claims. Never substitute a long training job for a missing unit test.

## Fair control matrix

At minimum, each approved pilot includes:

| ID | Method/control | Purpose |
|---|---|---|
| C0 | Raw-feature global Ridge | Matched analytic baseline on `h(x)`, with same `λ` policy. |
| C1 | FLY-CL current implementation | Fixed random expansion + WTA reference. |
| C2 | SOHO current implementation | OLDA/ETF + expansion/WTA reference; disclose replayed features. |
| C3 | T-SOHO strict low rank | Proposed graph code with `r<C_seen−1`. |
| C4 | T-SOHO random/label-independent orthonormal code | Isolates the contribution of confusion geometry. |
| C5 | T-SOHO shuffled/confounded graph | Falsifies graph-semantic explanation. |
| C6 | T-SOHO full-rank orthogonal and full-simplex controls | Detects change-of-basis/argmax-equivalence claims. |

Report a predeclared rank grid and `τ` grid selected from train-only data, then lock one configuration before reading test metrics. If a validation split changes the continual protocol, apply exactly the same split rule to every learned hyperparameter and control.

## Phase result template

Every completed phase must append a short result record (committed document or manifest) containing:

```text
phase:
git_commit:
worktree_status:
commands_exact:
exit_codes:
tests_passed/total:
environment:
backbone_and_checkpoint:
dataset_and_split:
class_order_hash:
seed(s):
config_hash:
state_inventory_and_bytes:
checkpoint_audit:
metrics:                 # blank for unit-only phases
gate_decision: pass|fail
known_deviations:
```

A failed gate blocks the next phase. Fixing it requires a new result record with the exact rerun command; it does not permit tuning against held-out test metrics.
