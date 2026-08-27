# ImageNet-R SOHO train-only optimization

Status: implementation prepared; Colab selection not started.

## Objective

Select the strongest SOHO replay configuration within a locked, affordable
search space and compare it fairly with original FLY fidelity. The study does
not select or discard random seeds by accuracy. All method comparisons use the
same class order, projection seed, frozen ViT checkpoint, train partition and
evaluation split.

The preceding diagnostic showed that current replay-wide GCV sometimes chooses
`0.1` or `1.0`, producing large transient AIA collapses. A fixed `1000`
counterfactual recovered `1.384` pp mean AIA and approximately matched FLY, but
that value was post-hoc. This new phase therefore selects Ridge strictly on
training validation before refining the SOHO representation.

## Locked protocol

- dataset: legacy processed ImageNet-R, train split only;
- backbone: frozen
  `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`;
- checkpoint SHA-256:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`;
- protocol: `configs/soho_imagenetr_optimal_train_only.json`;
- protocol SHA-256:
  `b1a5b2a819a30c35355540ca3ee3a9f96e3ad5e7672c27d444e8eac192f43004`;
- runner: `tools/soho_imagenetr_optimal_train_only.py`;
- runner SHA-256:
  `f333d010a4345744a12b8ea78a3bffc3abefcc0ab75dee9de403e13229cbf365`;
- notebook: `notebooks/soho_imagenetr_optimal_train_only_colab.ipynb`.

Selection is coordinate-wise rather than an infeasible claim of global
optimality:

1. at the previously selected representation `(density=0.2,
   coding_level=0.45)`, select fixed Ridge from
   `{10,100,1000,10000}`;
2. with that Ridge locked, search density `{0.1,0.2,0.3}` by coding level
   `{0.4,0.45,0.5}`;
3. confirm the selected SOHO against fixed original FLY
   `(synaptic_degree=300,coding_level=0.3,GCV exponents 6..9)` on untouched
   outer validation.

Development seed pairs are `2025/12025`, `3407/13407`, and `4421/14421`.
Outer confirmation adds `5501/15501` and `6619/16619`. The gate requires
positive mean paired AIA, at least four of five seed wins, and paired 95% CI
lower bound at least `-0.5` pp. A gate pass still requires review and does not
authorize test evaluation.

SOHO replay remains explicitly non-exemplar-free: it stores historical
frozen-backbone features and labels. This study optimizes fidelity accuracy; it
does not change that state contract.

## Local gate

```powershell
python -B -m pytest -q tests/test_soho_imagenetr_optimal_train_only.py tests/test_soho_imagenetr_gcv_diagnostic.py tests/test_cached_replay_baselines.py tests/test_soho_selfcontained.py
python -m json.tool configs/soho_imagenetr_optimal_train_only.json
python -m json.tool notebooks/soho_imagenetr_optimal_train_only_colab.ipynb
```

Focused result: `22 passed`, `19 warnings`, exit code `0`. The warnings are
existing PyTorch JIT deprecations and sparse CSC beta support.

Full repository regression command:

```powershell
python -B -m pytest -q
```

Result: `381 passed`, `20 warnings`, exit code `0`, in `57.73 s`.

Run the notebook from top to bottom on a Colab T4. It prints each candidate and
task, supports unit-level resume while the runtime storage survives, and
downloads `soho_imagenetr_optimal_train_only.zip`. Return that ZIP and stop for
audit; do not open the ImageNet-R test split.
