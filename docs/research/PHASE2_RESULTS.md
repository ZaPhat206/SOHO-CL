# Phase 2 — mathematical sanity-test results

Status: **PASS** for the scoped synthetic algebra tests. This is not an experiment result and does not validate T-SOHO accuracy, memory, or reproducibility on a dataset.

## Diff review

Reviewed Phase 2 addition: `tests/test_tsoho_math.py` only. It contains local synthetic helpers for one-hot labels, linear Ridge solve, positive Top-K WTA, and a simplex ETF; it imports no repository learner, dataset, backbone, SOHO, or FlyCL implementation. No model/experiment path was changed by Phase 2.

Review commands:

```bash
git -c safe.directory=D:/lab/FLY/SOHO-CL -C . status --short
git -c safe.directory=D:/lab/FLY/SOHO-CL -C . diff --check
```

`diff --check` completed without whitespace errors. The worktree also contains pre-existing/unrelated uncommitted work noted by audit (`notebooks/kaggle_runner.py`) and prior documentation additions; it was not interpreted as a Phase 2 code change.

## Exact test command and result

```bash
cd D:/lab/FLY/SOHO-CL
python -m pytest -q
```

Result: `6 passed in 7.10s`.

All tests set `torch.float64` explicitly (`DTYPE=torch.float64`) and use absolute tolerance `atol=1e-5`, relative tolerance `rtol=0`, unless an exact equality/assertion is stated. The synthetic seeds are 7, 11, 19, and 23 in the tests that use random samples; the Top-K counterexample and strict-rank projector test are deterministic constants.

## Per-test record

| Test | Result | What was checked | Mathematical conditions | Limitation |
|---|---|---|---|---|
| `test_streaming_G_and_Q_equal_batch_statistics` | PASS | Chunked `ΣXᵀX` and `ΣXᵀY` equal concatenated batch statistics. | Every sample is included once; the representation `X` and global one-hot class-column convention are fixed. | Does not prove correctness if features are changed after storage, classes are remapped inconsistently, or numeric accumulation is low precision/parallel-nondeterministic. |
| `test_streaming_ridge_logits_equal_batch_ridge_logits` | PASS | Ridge solved from streaming `G,Q` gives query logits equal to batch Ridge. | Same fixed features, labels, scalar `λ=0.37`, and positive-definite `G+λI`; exact linear solve. | Does not validate a GCV/λ policy, learned representation, dynamic transforms, or finite-precision GPU execution. |
| `test_orthogonal_transport_with_isotropic_ridge_preserves_logits` | PASS | `X'=XU`, `W'=UᵀW`, and logits agree. | `U` is a **full-dimensional square orthogonal** matrix (`UᵀU=I`); Ridge penalty is isotropic `λI`; both train and query features receive the same transform; the solve uses the same λ. | It does not apply to rank reduction, rectangular/non-orthogonal maps, anisotropic regularization, an added nonlinear operation, changed feature normalization, or an incorrectly transported classifier. It is not a claim that all transforms are no-ops. |
| `test_dynamic_topk_has_no_shared_linear_transport_in_general` | PASS | Two samples have identical old WTA output but distinct new WTA output, contradicting any common linear map. | Positive per-sample Top-1 WTA; old and new projections are the explicit matrices in the test; desired transport must be exact and sample-independent. | This only rules out **exact sample-independent linear transport in general**. It does not rule out approximate transport, a restricted data distribution, a sample-dependent/nonlinear map, or special projections where a transport happens to exist. |
| `test_full_rank_simplex_etf_preserves_raw_ridge_argmax` | PASS | Decoded full-simplex code Ridge and raw Ridge have the same argmax. | `E∈R^((C−1)×C)` satisfies `EEᵀ=I` and `EᵀE=I−11ᵀ/C`; Ridge uses identical `G,Q,λ`; all ETF columns have equal norm; decoder is `2xW_REᵀE−||e_c||²`; ties are absent or resolved identically. | It does not establish equivalence for strict low rank, non-simplex/non-equal-norm codes, a changed λ/objective, class-dependent decoder biases, ties with different tie rules, or a learned feature transform. |
| `test_strict_low_rank_code_has_expected_EtE_properties` | PASS | For `r=3<C−1=5`, `EEᵀ=I`; `EᵀE` is symmetric, idempotent, rank 3, and neither identity nor full centering projector. | Rows of `E` are orthonormal and rank is strictly below `C−1`. | It proves projector algebra only; it does not show that this code came from a useful confusion graph or improves any CL metric. |

## Research interpretation

The results support using `G,Q` as exact streaming sufficient statistics **only in a fixed feature space**, and they protect the novelty claim from two degeneracies: full-dimensional orthogonal reparameterization and full-rank simplex ETF argmax equivalence under the listed conditions. They also supply a concrete reason why the existing dynamic-OLDA/Top-K SOHO path requires replay/reprojection for exact updates.

They do not yet support an accuracy claim for T-SOHO, a claim that graph geometry helps, an exemplar-free checkpoint claim, or a FLY-CL reproduction claim. Those require later implementation, checkpoint audit, controls, and locked evaluation protocol.
