# SOHO self-contained Colab runbook

Use `notebooks/soho_selfcontained_final_colab.ipynb` on a Colab GPU. The
notebook evaluates the repository's existing SOHO method; it does not introduce
a new SOHO architecture.

## What is being evaluated

The authoritative cache-native method is `CachedSOHOReplayFidelity`. Existing
synthetic fidelity tests compare it task by task with `methods/sohocl.py`,
including OLDA/ETF updates, replay reprojection, GCV-selected Ridge values,
statistics, logits, predictions and checkpoint resume.

SOHO is not exemplar-free. After every task it retains all historical frozen
ViT features and labels, rebuilds the dynamic OLDA map, reprojects the complete
history through sparse projection and sample-dependent WTA, and refits the
global analytic classifier. The notebook counts those historical feature and
label tensors in persistent learner-state bytes and records their row counts.

The final comparison contains:

- current SOHO replay fidelity;
- current FLY-CL fidelity;
- streaming raw-feature Ridge.

All methods share the frozen ViT-B/16 checkpoint, preprocessing, task split,
class-order seed and replicate identity. Inference uses the global classifier
and does not receive a task ID.

## Experimental flow

1. Verify repository, runner, protocol and model-checkpoint hashes.
2. Audit CUB and ImageNet-R processed dataset identities.
3. Extract only official training features; `test.pt` must not exist.
4. Run synthetic correctness and original-vs-cache fidelity tests.
5. On official train only, form a deterministic stratified 80/20 outer split,
   then an 80/20 inner split of the development partition.
6. Select SOHO from the predeclared density/coding grid and select raw-Ridge
   lambda. SOHO and FLY retain their existing per-task GCV policies.
7. Record outer-validation confirmation and hash all selected settings.
8. Create an immutable authorization file. Only then materialize test features.
9. Refit every learner from empty state on the complete official train split
   and run six paired class-order/projection replicates.
10. Export metrics, 95% confidence intervals, curves, resource plots and a ZIP.

The final ZIP excludes feature caches and replay tensors. It contains evidence
and metrics, not a deployable SOHO checkpoint.

## Locked search spaces

- SOHO expansion width: 10,000;
- OLDA dimension: 768;
- ETF: enabled for every candidate;
- projection density: `{0.1, 0.2, 0.3, 0.5, 0.8}`;
- WTA coding level: `{0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.8}`;
- SOHO GCV exponents: `[-2, 10)`;
- raw-Ridge lambda: `10^-3` through `10^8`;
- split seed: `2025`;
- development replicate seeds: `(2025,4201)`, `(2026,4202)`, `(2027,4203)`;
- final replicate seeds: `(3031,5031)` through `(3036,5036)`.

The grid must not be changed after any test output is visible. If raw Ridge
selects an endpoint, the notebook stops before test so that the grid can be
reviewed as a new protocol rather than extended after seeing test results.

Selection is predeclared as a two-stage search. Stage 1 runs the five density
values at coding level `0.3` and the seven coding values at density `0.3` (11
unique configurations) on all three development replicates. Stage 2 takes the
top two density values and top three coding values by mean AIA and evaluates
their six-way Cartesian interaction on the same inner-validation protocol.
Previously evaluated interaction points are restored rather than rerun. The
maximum is 51 SOHO units per dataset instead of 105 for a full 35-by-3 search.

Among stage-2 candidates within `0.05` percentage point of the best mean AIA,
the protocol selects lower coding level first and then lower density. This
sparsity preference is locked before test evaluation. Coding level `0.8` and
density `0.8` are stress controls: they make WTA/projection nearly dense and
are not assumed to be good configurations.

## How to run

1. Push branch `experiment/soho-selfcontained` before opening Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Open the notebook and edit only repository/path values in cell 2.
4. Run cells in order. Each long unit prints `START`, one `TASK` line per stage,
   and `DONE`; completed units resume only when their context hash matches.
5. If selection returns anything other than `SELECTION_COMPLETE`, stop before
   the authorization/test cells.
6. Download `soho_selfcontained_three_dataset_results.zip` from the final cell.

Exact local correctness command:

```text
python -B -m pytest -q tests/test_soho_selfcontained.py tests/test_cached_replay_baselines.py
```

## Reporting limitations

- CIFAR-100, CUB and ImageNet-R test splits were already consumed by earlier
  repository phases. This protocol prevents new test-set tuning, but its test
  results are not first-use untouched held-out evidence.
- The processed ImageNet-R split has 19 duplicate content hashes across train
  and test, including 18 conflicting-label hashes. It must be labeled a legacy
  processed split, not a content-disjoint benchmark.
- Persistent learner state, peak allocated GPU memory and disk feature-cache
  bytes are different quantities. None may be substituted for another.
- A frozen feature cache is experiment infrastructure containing sample-level
  data. SOHO additionally retains sample-level feature history inside learner
  state; therefore no exemplar-free claim is permitted.
