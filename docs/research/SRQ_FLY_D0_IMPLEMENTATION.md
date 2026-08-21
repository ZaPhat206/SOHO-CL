# SRQ-FLY D0 implementation record

Status: implementation and synthetic correctness complete; real ImageNet-R D0
has not been run.

## Implemented scope

- deterministic groupwise-int8 strict-upper storage with exact float32 diagonal;
- float16 and int8 upper-triangular square-root storage;
- direct-int8 Gram, float16 square-root, and int8 SRQ-FLY learners;
- streaming `Q`, counts, global class mapping, analytic Cholesky/triangular solves,
  task-ID-free inference, checkpoint continuation, and persistent-state audit;
- strict five-task train-only runner with exact FLY-10k, exact FLY-4096, and
  raw-Ridge controls;
- resumable WTA/method artifacts and a Colab notebook with live progress;
- no changes to existing SOHO or FLY implementations.

The projected final 200-class SRQ-FLY state for `m=10000`, block size 256, and
group size 64 is 105,166,640 bytes, versus 452,006,952 bytes for matched exact
FLY under the same tensor-byte accounting (fraction 0.232666). This is a storage
projection, not an observed accuracy or runtime result.

## Verification

Environment used locally: Windows, CPU, repository Python/torch environment.
The expected sparse-CSC beta warning is inherited from `models/flyhash.py` and
does not indicate a failed assertion.

Exact commands and results:

```text
python -m pytest -q tests/test_srq_fly_math.py tests/test_srq_fly_learner.py tests/test_srq_fly_d0.py
19 passed, 20 warnings in 14.86s

python -m pytest -q
226 passed, 20 warnings in 36.56s

python -m json.tool notebooks\srq_fly_imagenetr_d0_colab.ipynb
PASS

python -m json.tool configs\srq_fly_imagenetr_d0_train_only.json
PASS

git diff --check
PASS
```

The synthetic runner was executed twice in the integration test. The second
execution loaded context-bound unit artifacts, establishing deterministic
resume behavior. A separate test confirmed that a visible `test.pt` causes the
runner to fail before loading training data or creating WTA caches.

## Claims and limitations

The tests establish storage determinism, exact stored diagonals, symmetric Gram
reconstruction, positive definiteness of a decoded triangular factor with
positive diagonal, low solve residual on synthetic problems, checkpoint
continuation, and absence of sample-shaped persistent learner tensors.

They do not establish that SRQ-FLY retains ImageNet-R accuracy, improves FLY,
or is Pareto-optimal. Groupwise int8 introduces cumulative approximation drift;
SPD-by-construction only prevents an indefinite decoded system. D0 must also
rule out the simpler exact FLY-4096 and direct-int8 Gram alternatives before a
square-root contribution can be justified.

## Next gate

After the branch is committed and pushed, run
`notebooks/srq_fly_imagenetr_d0_colab.ipynb` from top to bottom and return
`srq_fly_imagenetr_d0_train_only.zip`. Stop after the five-task train-validation
result. Do not expose `test.pt` and do not extend to 20 tasks unless D0 passes
and a separate protocol is reviewed.
