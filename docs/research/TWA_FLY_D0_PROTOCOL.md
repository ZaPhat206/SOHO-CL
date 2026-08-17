# TWA-FLY D0 — train-only complementarity protocol

## Purpose

Phase A falsified agreement-only TWA-FLY on the locked CIFAR-100 training
validation stream: its selected symmetric model changed one validation
prediction and gained only `0.0022` percentage point over matched FLY. D0 does
not tune or revive that method. It asks the cheaper prerequisite question for a
possible joint residual model: **does raw ViT Ridge correctly classify enough
examples that matched FLY misses?**

No held-out feature may be visible while D0 runs. Passing D0 authorizes review
of an implementation plan; it does not authorize CIFAR-100 test evaluation.

## Locked views and classifier fits

The protocol reuses the same frozen ViT features, fixed sparse FlyHash
projection, signed largest-value Top-K codes, class order, task split, and seed
as matched FLY. At every task it updates only

```
G_xx, G_zz, R_xz, Q_x, Q_z, counts.
```

Raw Ridge uses the independently locked Phase H value `lambda_x=0.01`. Matched
FLY retains its current-task GCV policy. Both classifiers use global class IDs
and predict without a task identifier.

For validation logits `l_x` and `l_z`, subtract the per-sample class-common
offset. Each view is divided by a training-only RMS computed exactly from its
Gram matrix and weights:

```
W_centered = W - mean_class(W)
s^2 = trace(W_centered^T G W_centered) / (N_train C).
```

The diagnostic fusion is

```
logits(alpha) = centered(l_z)/s_z + alpha * centered(l_x)/s_x.
```

`alpha=0` must be argmax-identical to matched FLY. Alpha is selected only from
the locked training-validation stream. This is a mechanism diagnostic, not a
reported held-out method result.

## Recorded evidence

For every cumulative validation stage, D0 records:

- raw, FLY, and oracle-union accuracy;
- both-correct, raw-only-correct, FLY-only-correct, and both-wrong counts;
- prediction disagreement and raw rescue rate among FLY errors;
- centered-logit correlation and RMS for both views;
- accuracy for every locked fusion alpha;
- raw/FLY Ridge coefficients and linear-system residuals.

Feature and WTA row caches remain disk experiment infrastructure. They are not
learner state and must not be serialized in a checkpoint.

## Projection/cache invariant

The WTA cache stores the exact sparse `projection.pt`; metadata records its
SHA-256, PyTorch materialization version, and a deterministic 16-row
re-encoding probe.
A legacy cache is upgraded only after current projection outputs reproduce its
cached active indices and values. Any projection hash or probe mismatch fails
closed; there is no silent regeneration or fallback.

## Gate

D0 passes only when all conditions hold:

1. projection/cache provenance probe passes;
2. raw Ridge equals the locked `0.01` protocol value;
3. validation average oracle-union headroom over FLY is at least `0.50` pp;
4. the best locked normalized fusion exceeds FLY by at least `0.10` pp;
5. every Ridge relative residual is at most `1e-5`;
6. `test.pt` remains physically hidden throughout the runner.

PASS decision: `REVIEW_JOINT_RESIDUAL_IMPLEMENTATION`.

FAIL decision: `STOP_TWO_VIEW_BRANCH`. Do not change seed, alpha grid, Ridge
policy, dataset, or gate after seeing the result. A new dataset or multi-seed
study requires a separately locked protocol.

## Exact local correctness command

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_twa_fly_diagnostic.py tests/test_twa_fly_pilot.py tests/test_twa_fly_math.py tests/test_twa_fly_learner.py
```
