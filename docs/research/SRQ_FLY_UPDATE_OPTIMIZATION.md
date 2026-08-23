# SRQ-FLY update optimization

Status: local correctness gate passed; dimension-matched CUDA timing pending.

## Scope and protocol integrity

The historical SRQ implementation is unchanged. Its locked source identities
remain:

- `methods/srq_fly/learner.py`:
  `5a1ee62022b98f6334c0c371087ffeb6512699f99c36fa4057db40c6219ac206`;
- `methods/srq_fly/storage.py`:
  `c095ecf98bdd991950998e41018c2b2a0ce13b7d16586cdac14143cea7c16c25`.

This is required because the completed held-out protocols bind both hashes.
The optimization lives under the explicit `methods.srq_fly_optimized`
namespace. It does not retroactively alter the reported three-dataset result.

The implementation is synthetic-only at this gate. It does not load a dataset,
checkpoint, feature cache, WTA cache, validation split, or test split.

## Changes

1. Equal-sized upper-triangular blocks are quantized in bounded batches of 16.
   Per-block group boundaries, scales, int8 values, diagonal precision, and
   logical persistent-state bytes remain unchanged.
2. Hot-path construction checks the finite source matrix once instead of
   forcing one CUDA-to-host synchronization per block. Checkpoint loading
   retains strict value validation.
3. The solver input is converted once, ridge is added directly to the diagonal,
   and the factor-error diagnostic uses a fused distance reduction.
4. `profile_updates=True` adds synchronized stage timings to transient
   diagnostics. Timing values are not checkpoint tensors.
5. The default `gram_cholesky` backend preserves the locked update equations.
   An explicit experimental `stacked_qr` backend computes the same square-root
   update from

   \[
   R_t = \operatorname{qr}_R\!\left(\begin{bmatrix}R_{t-1}\\H_t\end{bmatrix}\right),
   \qquad R_t^\top R_t=R_{t-1}^\top R_{t-1}+H_t^\top H_t.
   \]

   QR row signs are normalized so the upper-triangular diagonal is positive.
   The backend identity is checkpoint-locked. It is not the default until a
   dimension-10,000 CUDA benchmark passes.

## Invariants tested

- optimized float16/int8 storage is byte-identical to locked blockwise storage;
- optimized `gram_cholesky` resumes a locked checkpoint with identical logits,
  future weights, and persistent-state bytes;
- `stacked_qr` matches `gram_cholesky` factors and weights on a two-task
  synthetic stream;
- both backends remain structurally SPD and satisfy the solver tolerance;
- profiling metadata is not persistent learner state;
- the benchmark uses seed 2025 and contains no held-out mode;
- every historical source-identity test still passes.

## Exact commands and local results

Environment: Python 3.13.5, PyTorch 2.12.0+cpu, Windows CPU.

```text
python -m pytest -q tests/test_srq_fly_optimized.py
8 passed, 1 warning in 9.63s

python tools/srq_fly_update_benchmark.py --config configs/srq_fly_update_optimization_smoke.json --output "$env:TEMP\srq_fly_update_optimization_smoke.json" --device cpu
status=pass
optimized_gram relative logit drift=0
optimized_qr relative logit drift=8.287693731290346e-08
locked/optimized persistent state=147568 bytes for every backend

python -m pytest -q
293 passed, 20 warnings in 64.27s

python -m json.tool configs/srq_fly_update_optimization_smoke.json
PASS
python -m json.tool configs/srq_fly_update_optimization_fly10000.json
PASS
python -m json.tool notebooks/srq_fly_update_optimization_colab.ipynb
PASS
git diff --check
PASS
```

The final CPU smoke observed 0.02367 s for locked SRQ, 0.01479 s for optimized
`gram_cholesky`, and 0.01841 s for `stacked_qr`. These are two-task diagnostic
wall times with warm-up/order noise. They are not a speed claim and are not
comparable to paper runtime.

The sparse-CSC beta warning is inherited from the unchanged FLY projection.

## CUDA handoff gate

Use `notebooks/srq_fly_update_optimization_colab.ipynb` after the branch is
committed and pushed. The notebook first runs source-identity/correctness tests,
then the small CUDA smoke, then the dimension-matched `m=10000` synthetic gate.
It prints one `TASK` line per stage and exports
`srq_fly_update_optimization.zip`.

The CUDA evidence must determine the backend; no backend is selected from the
CPU smoke. A later dataset run requires a separate locked train-only protocol.

## Limitations

- This phase establishes implementation parity, not accuracy or peak-memory
  improvement.
- The benchmark runs methods sequentially in one process and does not support a
  paper claim about absolute runtime or peak VRAM.
- `stacked_qr` may be slower or require more temporary memory at dimension
  10,000; it remains experimental until measured.
- Feature extraction is absent, so this phase says nothing about end-to-end
  training time.
