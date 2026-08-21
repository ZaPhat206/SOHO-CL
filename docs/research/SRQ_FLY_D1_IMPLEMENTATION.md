# SRQ-FLY D1 implementation record

Status: 20-task train-only runner and Colab notebook implemented; real D1 has
not been run.

## Scope

D1 preserves every D0 method and hyperparameter and extends the locked
ImageNet-R training-validation stream from 5 to 20 tasks. Exact FLY-10000 and
SRQ-FLY are evaluated in one paired unit so prediction agreement and relative
logit drift are reduced online without writing per-sample predictions.

The implementation adds no held-out mode. It fails before loading training data
when `test.pt` is visible. It reuses the verified D0 train, FLY-10000 WTA, and
FLY-4096 WTA caches as experiment infrastructure; these caches are not learner
state.

## Verification

Exact local commands and results:

```text
python -m pytest -q tests/test_srq_fly_math.py tests/test_srq_fly_learner.py tests/test_srq_fly_d0.py tests/test_srq_fly_d1.py
22 passed, 20 warnings in 10.99s

python -m pytest -q
229 passed, 20 warnings in 35.44s

python -m json.tool configs\srq_fly_imagenetr_d1_train_only.json
PASS

python -m json.tool notebooks\srq_fly_imagenetr_d1_colab.ipynb
PASS

git diff --check
PASS
```

The sparse-CSC beta warning is inherited from the unchanged FLY projection.
The synthetic D1 runner was executed twice to verify context-bound unit resume,
and a separate test verified refusal of a visible held-out cache.

## Limitations

Synthetic correctness does not establish long-horizon accuracy. D1 uses one
predeclared seed and training validation only. Timing fields are diagnostic wall
times and are not suitable for paper runtime comparison because the protocol
does not provide synchronized hardware benchmarking. Only an audited D1
artifact can authorize a later multi-seed protocol.

## Colab handoff

After commit and push of `feature/srq-fly-d1`, run
`notebooks/srq_fly_imagenetr_d1_colab.ipynb` cells 1 through 7 and return
`srq_fly_imagenetr_d1_train_only.zip`. Stop without evaluating held-out data.
