# SRQ-FLY D3 implementation record

The D3 implementation is isolated from existing SOHO and FLY learners. It adds
one explicit configuration, one train-only runner, synthetic tests, a Colab
notebook, and a runbook. The existing `SquareRootFLYLearner`, exact-FLY solver,
WTA cache implementation, frozen backbone, and dataset loaders are reused
without modification.

Key safeguards implemented by `tools/srq_fly_d3_cub.py` are:

- strict config keys, duplicate-key rejection, seed and dataset identity lock;
- exact verification of feature metadata, finite values, class coverage, and
  absence of `test.pt`;
- disjoint nested split verification and hashes for all four partitions;
- exact seeded projection-prefix verification while explicitly recomputing
  dimension-specific Top-K codes;
- independent inner selection for FLY-10,000, FLY-4,518, and raw Ridge;
- shared selected lambda for paired exact/SRQ FLY-10,000;
- analytic and runtime persistent-state agreement;
- resumable units bound to source/config/split/code/projection hashes;
- result contracts that reject sample/feature/code/history fields.

Implementation gate command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_srq_fly_d3_cub.py
```

Focused result: `6 passed`. Expanded SRQ/WTA/CUB suite: `56 passed`. Full
repository command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q
```

Result: `247 passed` in `17.77s`. Warnings were the existing PyTorch JIT
deprecation and sparse-CSC beta/invariant warnings; none was suppressed.

The large CUB study is intentionally not run locally. After the complete test
suite and diff review pass, the user action is to run
`notebooks/srq_fly_cub_d3_train_only_colab.ipynb` and return the generated ZIP.
