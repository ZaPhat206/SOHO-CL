# MT-SOHO Phase 1A implementation record

Date: 2026-08-27. This record covers implementation and synthetic correctness
only. No CIFAR-100 test feature or test label was opened, and no empirical
accuracy claim is made yet.

## Implemented components

- `methods/mt_soho/statistics.py`: minimal fixed-anchor and raw-view sufficient
  statistics; no cross-view or sample-level tensor is retained.
- `methods/mt_soho/geometry.py`: pooled within-scatter reconstruction,
  shrinkage whitening, deterministic low-rank class targets, shuffled control,
  Cholesky solves, and exact moment transport.
- `methods/mt_soho/learner.py`: fixed sparse WTA anchor, global anchor Ridge,
  post-WTA analytic transport, transported low-rank Ridge head, checkpoint and
  persistent-state audit.
- `tools/mt_soho_phase1.py`: resumable nested train-only selection and outer
  validation runner which fails when `test.pt` is visible.
- `configs/mt_soho_phase1a_cifar100_train_only.json`: locked Phase 1A search,
  seeds, controls, and gates.
- `notebooks/mt_soho_phase1a_cifar100_colab.ipynb`: Colab setup, correctness
  gate, train-only cache restore/extraction, live experiment progress, summary,
  and evidence download.

Legacy SOHO, FLY, SRQ-FLY, and the uncommitted SRQ-SOHO exploratory files were
not modified.

## Correctness results

Command:

```text
python -m pytest -q tests/test_mt_soho.py tests/test_mt_soho_phase1.py
```

Result: `12 passed, 1 warning in 6.41s`. The warning is PyTorch's documented
sparse-CSC beta warning. Tests establish:

1. streaming fixed-anchor `G,Q` equal batch moments;
2. raw moments exactly reconstruct pooled within-class scatter;
3. transported moments equal explicit batch reprojection;
4. streaming logits equal the explicit batch oracle after the transport
   changes;
5. target construction is deterministic, finite, and rank-bounded by `C-1`;
6. shuffled controls share identical anchor statistics but change targets;
7. sample-free checkpoint round-trip preserves logits;
8. no persistent tensor has a historical sample-count dimension;
9. inference has no `task_id` argument;
10. train-only runner refuses a visible `test.pt`, executes nested splits, and
    resumes completed units.

Full repository command:

```text
python -m pytest -q
```

Result: `372 passed, 20 warnings in 56.78s`.

Notebook JSON and every code cell were parsed successfully with Python's JSON
and AST parsers.

## Interpretation gate

The mathematical/software gate passes. This does **not** show that MT-SOHO is
more accurate than FLY. The next authorized action is the locked CIFAR-100
train-only notebook. A Phase 1A accuracy failure stops this branch; a pass only
authorizes matched-width comparison with legacy replay SOHO before any held-out
evaluation or SRQ integration.
