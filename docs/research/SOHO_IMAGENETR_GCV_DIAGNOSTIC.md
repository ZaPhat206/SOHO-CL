# SOHO ImageNet-R GCV diagnostic

Status: implementation prepared; real Colab run not started.

Implementation source identities:

- protocol SHA-256: `83e11107a531ec8be49f2e32f9e40151e78ba621777894ae3baf7dc6858eb468`;
- runner SHA-256: `38c4fa0d80367bcd31a25bd511d514ef1202032f2553b0210a430acb8241e06c`.

## Why this phase exists

The locked artifact `soho_selfcontained_three_dataset_results.zip` (SHA-256
`f3df4606644f4660e78b637a58f198b7ef345a84d8f6f222f584e9a888ea7167`)
already contains the requested comparison:

- selected SOHO replay: density `0.2`, coding level `0.45`, ETF enabled;
- original/fidelity FLY: synaptic degree `300`, coding level `0.3`;
- six paired final replicates `(3031,5031)` through `(3036,5036)`.

SOHO was selected on training-only inner validation. FLY was deliberately the
fixed original/fidelity control, not validation-tuned. The artifact therefore
is the correct SOHO-tuned versus original-FLY comparison.

Its ImageNet-R test AIA gap is not a final-stage collapse. SOHO and original
FLY finish at `71.9783` and `71.9477`, respectively, while their AIA values are
`76.0932` and `78.2156`. Mean SOHO-minus-FLY stage accuracy reaches its largest
negative gaps at tasks 7 (`-6.835` pp), 9 (`-6.132` pp), and 8 (`-4.363` pp),
then recovers to `+0.031` pp at task 20.

Across all 120 replicate/stage observations in the consumed artifact, SOHO's
selected Ridge coefficient is strongly associated with the collapse:

| Selected Ridge | observations | mean SOHO-FLY stage gap |
|---:|---:|---:|
| `0.1` | 20 | `-6.661` pp |
| `1` | 22 | `-5.172` pp |
| `100` | 9 | `-0.206` pp |
| `1000` | 69 | `-0.084` pp |

This is observational and task-confounded; it does not prove causality. The
new train-only diagnostic evaluates a fixed-`1000` counterfactual from exactly
the same per-stage SOHO `G,Q,R,W` state as current GCV. It also records OLDA
projection drift, WTA support turnover and code cosine on a deterministic
training probe. No held-out feature is loaded.

## Locked run

- config: `configs/soho_imagenetr_gcv_diagnostic.json`;
- runner: `tools/soho_imagenetr_gcv_diagnostic.py`;
- notebook: `notebooks/soho_imagenetr_gcv_diagnostic_colab.ipynb`;
- default diagnostic/split seed: `2025`;
- paired class/projection seeds: exact historical six pairs `3031/5031` through
  `3036/5036`;
- validation: deterministic stratified 20% of official ImageNet-R training;
- methods: current selected SOHO GCV, post-hoc fixed-1000 SOHO counterfactual,
  and original FLY fidelity.

The fixed-1000 row is explicitly post-hoc because the value was motivated by
an already consumed artifact. Even if it wins, it is not a newly validated
method and cannot authorize test evaluation. A positive diagnostic must be
followed by a preregistered policy tested on new train-only seeds or a new
dataset.

SOHO remains non-exemplar-free: historical frozen-backbone features and labels
are replay state and are counted. ImageNet-R remains the disclosed legacy
processed split with 19 cross-split duplicate hashes, including 18 conflicting
labels. This phase never creates or reads `test.pt`.

## Required local checks

```powershell
python -B -m pytest -q tests/test_soho_imagenetr_gcv_diagnostic.py tests/test_cached_replay_baselines.py tests/test_soho_selfcontained.py
python -m json.tool configs/soho_imagenetr_gcv_diagnostic.json
python -m json.tool notebooks/soho_imagenetr_gcv_diagnostic_colab.ipynb
```

After the Colab run, return `soho_imagenetr_gcv_diagnostic_train_only.zip` and
stop for audit.

## Local implementation gate

Exact command:

```powershell
python -B -m pytest -q
```

Result on the implementation worktree: `376 passed`, `20 warnings`, exit code
`0`, in `72.75 s`. The warnings are pre-existing PyTorch JIT deprecations,
sparse CSC beta support and disabled sparse invariant checks. The focused gate
also passed with `17 passed`, `19 warnings`:

```powershell
python -B -m pytest -q tests/test_soho_imagenetr_gcv_diagnostic.py tests/test_cached_replay_baselines.py tests/test_soho_selfcontained.py
```

The real diagnostic is intentionally not run locally: it needs the verified
ImageNet-R training-feature cache and a CUDA runtime. Run the notebook from top
to bottom and return its ZIP before any new method or held-out evaluation is
authorized.
