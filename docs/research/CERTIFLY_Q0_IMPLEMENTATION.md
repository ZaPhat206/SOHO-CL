# CertiFLY Q0 implementation result

Date: 2026-08-20. Branch: `feature/certifly`. Base commit before the Q0
working tree: `3f5b6debd4ff1c97a00fb4f99e4dd125cc493a8d`.

## Scope completed

Q0 implements only the mathematical and learner-state prototype required to
decide whether a train-only ImageNet-R feasibility run is justified:

- exact FLY sparse projection and dynamic WTA semantics are preserved;
- the full-coordinate Gram diagonal is stored exactly;
- strict-upper normalized-correlation blocks use deterministic symmetric int8
  quantization, with deterministic int16 promotion when allowed;
- cumulative streaming quantization error is bounded by a
  Frobenius/triangle-inequality certificate;
- the analytic Ridge classifier uses Cholesky solves and rejects a state when
  the certificate cannot guarantee positive definiteness;
- learner checkpoints contain projection, quantized Gram, `Q`, counts, class
  IDs, classifier, and scalar metadata, but no images, historical features, or
  historical WTA codes;
- prediction is global over all seen classes and has no `task_id` argument;
- failed task updates are transactional and restore the previous valid state;
- a locked, resumable train-only Q1 runner and Colab notebook are present, but
  no ImageNet-R Q1 experiment was run locally.

Existing SOHO and FLY implementations were not modified.

## Mathematical gates

Synthetic tests use deterministic seeds `301`, `307`, `311`, `401`, `409`,
and `47`. Core quantization/solver tests use `torch.float64`; the learner also
tests the repository FLY feature path. Assertions cover:

1. deterministic block quantization, symmetry, and an exact Gram diagonal;
2. measured spectral error no larger than the cumulative certificate;
3. adaptive int8-to-int16 promotion under a tighter error budget;
4. certified SPD, solver residual, classifier perturbation, per-sample logit
   perturbation, and argmax preservation when the certified margin condition
   holds;
5. projected all-int8 ImageNet-R learner state below 25% of matched exact FLY;
6. exemplar-free persistent-state structure, global prediction, WTA sparsity,
   checkpoint/resume, mismatch rejection, and transactional failure;
7. locked Q1 runner behavior, deterministic resume, hidden-test refusal,
   strict config parsing, and mandatory seed `2025`.

The analytical all-int8 projection for `d=768`, `H=10,000`, `p=300`,
`C=200`, block size `256`, and float32 statistics is:

- compressed persistent tensor state: `102,045,232` bytes;
- matched dense-Gram state estimate: `452,006,952` bytes;
- fraction: `0.2257603153`.

This is a tensor-state projection, not a measured checkpoint-file size. Python
and serialization metadata are excluded consistently from both sides.

## Exact commands and results

Environment used for local Q0 verification:

- Python `3.13.5`;
- PyTorch `2.12.0+cpu`;
- device: CPU.

Targeted CertiFLY suite:

```powershell
python -m pytest -q tests\test_certifly_math.py tests\test_certifly_learner.py tests\test_certifly_q1.py
```

Result after final Q0 fixes: `14 passed`, with only the repository's existing
PyTorch sparse-CSC beta/invariant warnings and PyTorch JIT deprecation warnings.

Final full repository suite:

```powershell
python -m pytest -q
```

Result: `207 passed`, same warning classes.

Notebook structural validation:

```powershell
@'
import ast, json
from pathlib import Path
path = Path('notebooks/certifly_imagenetr_train_only_colab.ipynb')
notebook = json.loads(path.read_text(encoding='utf-8'))
for index, cell in enumerate(notebook['cells'], 1):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']), filename=f'cell-{index}')
print('notebook_json=PASS')
'@ | python -
```

Result: valid notebook JSON; all ten cells and all code-cell syntax passed.

## Interpretation and limitations

Q0 establishes correctness of the implementation and the stated deterministic
perturbation bounds on synthetic problems. It does **not** establish that the
ImageNet-R state will remain below 25%, that enough blocks can stay int8, that
accuracy will be within 0.5 percentage point of exact FLY, or that runtime will
improve. Quantization keeps all coordinates but still reconstructs a dense Gram
for the prototype solve, so Q0 claims persistent-state compression only—not
peak runtime-memory or solve-time reduction.

The cumulative certificate is deliberately conservative: it adds local
Frobenius bounds over streaming requantizations. A Q1 failure can therefore
mean either that the storage/accuracy hypothesis is false or that this bound is
too loose; the report must distinguish those cases using the recorded
`error/ridge` trajectory and int8/int16 block histogram.

The Q1 feature and WTA caches contain sample-level training data. They are
experiment infrastructure on disk and are excluded from the learner state.
Only a checkpoint produced from `CertiFLYLearner.state_dict()` qualifies for
the exemplar-free claim.
