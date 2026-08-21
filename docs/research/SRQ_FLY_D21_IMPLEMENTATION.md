# SRQ-FLY D2.1 implementation record

Status: nested train-only runner implemented; real ImageNet-R D2.1 has not
been run.

## Scope

D2.1 adds no learner and changes neither original FLY nor SRQ-FLY. It reuses
the exact FLY-4518 evaluator and D2 WTA cache. Six predeclared lambdas are
selected using only a deterministic inner split of D1 outer-training data.
Only the selected lambda is subsequently evaluated on outer validation.

The runner rejects a visible `test.pt`, a modified D2 artifact, mismatched
train/split/projection identities, duplicate config keys, an unlocked lambda
grid, stale resume units, and incorrect analytic/runtime state accounting.
No sample indices, predictions, features, labels, or WTA codes are serialized
to learner or evidence state.

## Verification

```text
python -m pytest -q tests/test_srq_fly_math.py tests/test_srq_fly_d2_state_match.py tests/test_srq_fly_d21_lambda_robustness.py
18 passed, 20 warnings in 17.74s

python -m pytest -q
241 passed, 20 warnings in 40.71s

python -m json.tool configs\srq_fly_imagenetr_d21_lambda_robustness.json
PASS

python -m json.tool notebooks\srq_fly_imagenetr_d21_lambda_robustness_colab.ipynb
PASS

git diff --check
PASS
```

The sparse-CSC warnings come from unchanged FLY code. Tests cover nested split
disjointness and coverage, absence of outer-validation selection leakage,
deterministic tie-breaking, strict D2 identity, visible-test refusal, config
validation, exact-result/sample-state rejection, canonical resume artifacts,
analytic/runtime state agreement, and end-to-end synthetic rerun equality.

The locked config SHA-256 is
`3c5b54ffedacf5620c8cd9123acb187f5cbf958023b37aebecf9f00c45f73e96`
and is bound identically in the notebook and runbook.

## Handoff

After commit and push of `feature/srq-fly-d21-lambda-robustness`, run cells 1
through 7 of
`notebooks/srq_fly_imagenetr_d21_lambda_robustness_colab.ipynb`. Return
`srq_fly_imagenetr_d21_lambda_robustness.zip` and stop without held-out
evaluation.
