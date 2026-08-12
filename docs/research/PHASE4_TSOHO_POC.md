# Phase 4 T-SOHO POC

Status: **Implementation ready; full experiment pending Kaggle execution.** No full experiment has run.

Implemented methods: `raw_ridge`, `random_orthogonal_code`, `truncated_simplex_code`, and `spectral_confusion_code` (T-SOHO). All operate on fixed frozen `(B,768)` features with bounded sufficient state (`G,Q,counts,sums,sq_sums`, code/projector/weights and metadata), without replay or Task-ID inference. Code methods use `E:(r,C_seen)` with strict `r<C_seen-1` (implemented as `r≤C_seen−2`); with fewer than three seen classes they use documented raw-Ridge fallback. After a task, `E` is rebuilt from all retained class statistics and `P` is resolved. `raw_ridge` uses `E=I_C` and the same nearest-code scorer; its argmax is equivalent to conventional raw Ridge.

Core formulation: `W=solve(G+lambda I,Q)`. Code methods use `P=solve(G+lambda I,QE^T)` and `2(XP)E-||E||^2`. Spectral affinity uses training statistics only, pooled diagonal variance, median finite off-diagonal Mahalanobis bandwidth, and largest non-trivial Laplacian eigenvectors. Eigenvector signs are canonicalized; degenerate eigenspaces are not claimed unique.

Local validation:

```bash
python -m pytest -q tests/test_tsoho_learner.py tests/test_tsoho_math.py tests/test_backbone_checkpoint.py tests/test_streaming_raw_ridge.py
python tools/experiment_runner.py --tiny-synthetic --feature-cache-dir %TEMP%\tsoho_cache --output-dir %TEMP%\tsoho_output --method spectral_confusion_code --rank 2 --ridge-lambda 0.5 --seed 1993
```

Fairness: all internal controls consume the same input features, class order, lambda, rank (except raw Ridge), seed and nearest-code scorer. Feature caches are experiment infrastructure, never learner state; persistent-state bytes exclude backbone/checkpoint/cache/results. Full-scale extraction/matrix runs are intentionally deferred to Kaggle.

The runner writes/validates a cache metadata schema, supports `--extract-features-only`, writes the requested run artifacts, and checkpoints a learner state after every task to `progress.pt` for `--resume` (deleted on a completed run). Full cache extraction/matrix runs are intended for Kaggle GPU, not local CPU.

Known limitations: only synthetic tests have run locally; no accuracy or SOTA claim is made; spectral eigenvectors may rotate in degenerate eigenspaces; Kaggle compatibility remains to be executed on the actual selected inputs.
