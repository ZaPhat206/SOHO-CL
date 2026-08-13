# Phase H — locked CIFAR-100 multi-seed results

Status: **complete and frozen**. The held-out CIFAR-100 artifact passed the
integrity audit. CIFAR-100 must not be used for further hyperparameter tuning.

## Artifact and protocol identity

- Artifact: `phaseh_multiseed_results.zip`
- ZIP bytes: `121181`
- ZIP SHA-256:
  `b4ced72abed8953004227008b9a82d8bf7de243b6b3475094ddab0ece0079b73`
- Locked Phase H manifest SHA-256:
  `cdfb716707215aa9f2101b1a49f833a1f547706d50b30ac4a774625ae63c6495`
- Phase G evidence ZIP SHA-256:
  `9ecaa259deb998f36abdd8052145b17a0ce84adeeb2168b29a83c039868cbc77`
- Frozen train-feature cache SHA-256:
  `b1421472ecc9054cf8fa7756f91f89c80e6b231d5a869389d353debe4f5098ab`
- Checkpoint SHA-256:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`
- Runtime: Python `3.12.13`, PyTorch `2.11.0+cu128`, CUDA device.
- Seeds: `1993, 2025, 3407, 4421, 5501`; five distinct class orders.
- Grid completeness: `40/40` unique seed/method units.
- Held-out test opened: yes, only by the locked evaluation runner.
- Test-time hyperparameter search: no.

The ZIP CRC passed, contained no duplicate members, and every result had the
locked manifest/cache identity, complete ten-stage accuracy matrix, finite
metrics, and the correct exemplar disclosure. The artifact records the commit
that created the feature cache but not the exact evaluation-runner commit.
Notebook output and branch history identify the intended runner as `7889eb0`;
the missing commit field remains a provenance limitation and is fixed before
the next dataset.

## Results

All accuracy values are percentage points and are mean ± sample standard
deviation over the five locked class orders.

| Method | Average incremental accuracy | Final accuracy | Forgetting | Persistent state | Exemplar-free |
|---|---:|---:|---:|---:|:---:|
| FLY-CL | 92.8509 ± 0.3317 | 88.7400 ± 0.0552 | 4.5022 ± 0.3025 | 444,006,550 B | yes |
| current SOHO replay | 92.7383 ± 0.3905 | 89.0500 ± 0.0900 | 4.0533 ± 0.2423 | 597,748,880 B | **no** |
| full raw residual | 92.2783 ± 0.3632 | 87.6300 ± 0.1000 | 5.0378 ± 0.3327 | 20,330,904 B | yes |
| Schur residual, rank 64 | 92.2007 ± 0.3458 | 87.2820 ± 0.1725 | 5.3356 ± 0.4226 | 15,003,032 B | yes |
| raw Ridge | 92.0413 ± 0.3457 | 87.1500 ± 0.0000 | 5.2956 ± 0.3231 | 2,974,096 B | yes |
| Fisher residual | 91.8819 ± 0.3303 | 86.5040 ± 0.1090 | 5.8844 ± 0.3864 | 15,003,032 B | yes |
| random residual | 91.0595 ± 0.3321 | 85.9340 ± 0.1609 | 5.6044 ± 0.4199 | 15,003,032 B | yes |
| anchor only | 90.8851 ± 0.3204 | 85.6240 ± 0.2034 | 5.6822 ± 0.5218 | 14,518,680 B | yes |

The maximal Schur relative solver residual was `4.5903e-6`, below the locked
`1e-4` tolerance. Its final retained-correction energy ranged from `0.8069` to
`1.0` as the effective rank grew with the number of seen classes.

## Preregistered paired conclusions

Paired 95% intervals use the two-sided Student-t critical value for four
degrees of freedom.

| Comparison (Schur minus control) | Metric | Mean difference | Paired 95% CI | Conclusion |
|---|---|---:|---:|---|
| raw Ridge | average incremental accuracy | +0.1594 | [+0.1054, +0.2134] | small, consistent gain |
| raw Ridge | final accuracy | +0.1320 | [-0.0822, +0.3462] | inconclusive |
| raw Ridge | forgetting | +0.0400 | [-0.2408, +0.3208] | no established improvement |
| Fisher residual | average incremental accuracy | +0.3188 | [+0.2603, +0.3774] | Schur wins |
| Fisher residual | final accuracy | +0.7780 | [+0.6350, +0.9210] | Schur wins |
| random residual | average incremental accuracy | +1.1412 | [+0.9685, +1.3139] | Schur wins |
| random residual | final accuracy | +1.3480 | [+1.2050, +1.4910] | Schur wins |
| full raw residual | average incremental accuracy | -0.0776 | [-0.1066, -0.0485] | full residual wins |
| full raw residual | final accuracy | -0.3480 | [-0.4629, -0.2331] | full residual wins |

Thus the Schur direction rule is not equivalent to an arbitrary or standard
Fisher rank-64 residual. It improves average incremental accuracy over raw
Ridge on all five orders, but the effect is small and does not establish a
final-accuracy improvement. Full residual remains better and only costs about
`5.33 MB` more persistent state.

## Exploratory efficiency comparisons

FLY/SOHO paired intervals were not included in the preregistered comparison
list, so the following are explicitly post-hoc descriptive results.

- Schur trails FLY by `0.6502` average-incremental points and `1.4580` final
  points, while using about `29.6×` less persistent state and `19×` less peak
  PyTorch CUDA allocated memory.
- Schur trails replay SOHO by `0.5376` average-incremental points and `1.7680`
  final points. SOHO retains historical sample features/labels and is not
  exemplar-free.
- Cached-classifier update time averaged `1.22 s` for Schur, `0.21 s` for raw
  Ridge, `219.53 s` for FLY, and `65.20 s` for replay SOHO. These timings omit
  feature extraction and are not comparable to paper hardware/runtime values.

FLY seed 1993 obtained `93.1061` average incremental accuracy, `0.7839` below
the external paper value `93.89`. Per protocol amendment 1 this is a diagnostic,
not a stopping gate. This run must not be called a strict paper reproduction.

## Frozen interpretation

Phase H supports only the following claim:

> With the locked frozen ViT representation and matched residual controls,
> Schur selection yields a small but consistent average-incremental improvement
> over raw Ridge and clearly outperforms Fisher/random direction selection,
> while remaining exemplar-free and far smaller than FLY/replay SOHO.

It does not support claims of superior final accuracy over raw Ridge, superior
accuracy over FLY/SOHO, global hyperparameter optimality, or paper-ready
cross-dataset generalization. The next dataset must be preregistered before its
test split is opened. CIFAR-100 results must not be used to tune another
CIFAR-100 configuration.
