# SRQ generalization M6 width-sweep runbook

Status: formal gate failure recorded; diagnostic sweep complete. M6 does not
authorize held-out test evaluation or post-hoc width selection.

## Question

M6 measures how accuracy, persistent state, and analytic-update time scale as
the RanPAC random-ReLU expansion width grows. It does not search for an
accuracy-optimal width. The locked widths are `2,000`, `4,000`, `6,000`,
`8,000`, `10,000`, `15,000`, and `20,000`, declared before the run.

At every width, three backends receive exactly the same projected features:

1. Exact FP32 Gram;
2. FP16 square-root;
3. frozen P2B mixed INT8/FP32.

All projections are prefixes of one standard-normal 768-by-20,000 projection
generated with seed 2025. The class order, task split, outer validation split,
and calibration indices are identical across widths. Each width selects one
Ridge coefficient on the disjoint train-only calibration partition, then
shares that coefficient across its three backends.

## State accounting

For every method and width, total persistent bytes include the dense random
projection, quadratic Gram or compressed factor, class counts, cross
statistic, and classifier. The result separately records projection bytes and
quadratic/factor bytes. Runtime feature caches are excluded from learner state
and are not packaged.

The sweep reports log--log slopes for total state. Because total state includes
linear projection and classifier terms as well as the quadratic state, its
finite-range slope need not equal exactly two. The quadratic/factor byte
columns expose the remaining asymptotic quadratic dependence directly.

## Gates

- all seven predeclared widths complete;
- projection prefixes remain nested;
- total state grows strictly with width for every backend;
- P2B state is smaller than Exact state at every width;
- FP16 loses at most 0.05 validation-AIA points to Exact at each width;
- P2B loses at most 0.25 validation-AIA points to Exact at each width;
- every solver relative residual is at most `1e-5`.

A scientific gate failure is retained in the artifact and cannot be overridden
with test accuracy. Non-monotone accuracy across width is reported rather than
treated as failure because finite validation performance is not mathematically
required to increase monotonically with random-feature dimension.

## Expected output

The Colab notebook exports a ZIP containing:

- `config.json`;
- `m6_results.json`;
- `width_sweep.csv`;
- `width_sweep_accuracy_state.svg`;
- `width_sweep_update_time.svg`.

Both figures are vector graphics generated directly from the locked JSON. M6
remains one controlled train-only development seed and does not prove scaling
on other datasets, backbones, hardware, or analytic frontends.

To reduce GPU peak at widths 15k and 20k, Exact, FP16, and P2B execute
sequentially at each width. They still use the same projection prefix, Ridge
coefficient, training partition, and validation partition. Analytic-update
time excludes repeated representation encoding. Sequential execution changes
only scheduling, not the fitted systems.

## Recorded result

Artifact: `srq_generalization_m6_width_sweep_train_only.zip`, SHA-256
`b2739b9da023ebd2eedb6fdfe01c394e94f252773e847533b35350021c3d239e`.
Result JSON SHA-256:
`648630c5f0b70ed85e675942a8c2fb10e6f33eb3ff8f03e8a71421de22f44cbb`.
The archive passes CRC, contains exactly the five expected files, embeds the
byte-identical locked config, reports clean source commit `d96714e`, and has
`uses_test_set=false`.

| Width | Exact AIA | FP16 AIA | P2B AIA | Exact--P2B (pp) | Exact/P2B state (MiB) | P2B reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 91.1474 | 91.1477 | 91.1536 | -0.0062 | 22.64 / 9.42 | 58.4% |
| 4,000 | 91.9818 | 91.9718 | 91.9750 | 0.0068 | 75.81 / 22.89 | 69.8% |
| 6,000 | 92.1427 | 92.1460 | 92.1335 | 0.0093 | 159.49 / 40.42 | 74.7% |
| 8,000 | 92.3000 | 92.3049 | 92.2378 | 0.0622 | 273.68 / 61.99 | 77.3% |
| 10,000 | 92.4428 | 92.4435 | 92.3551 | 0.0877 | 418.40 / 87.62 | 79.1% |
| 15,000 | 92.6220 | 92.6210 | 92.4261 | 0.1960 | 913.70 / 169.43 | 81.5% |
| 20,000 | 92.6429 | 92.6439 | 92.3870 | **0.2558** | 1,599.73 / 276.58 | 82.7% |

Six of seven gates pass. Only P2B accuracy retention fails: the 20,000-width
loss exceeds the locked 0.25-point limit by 0.005845 points. This remains a
formal failure even though it is close to the boundary. The maximum FP16 loss
is 0.0100 points and the maximum solver residual is `5.83e-6`; there is no
numerical failure. Thus, the evidence supports stable square-root/QR updates
but shows a width-dependent INT8 approximation cost that requires M7
diagnosis.

The total-state log--log slopes are 1.853 for Exact, 1.613 for FP16, and 1.470
for P2B over this finite range. They are below two because total state also
contains linear projection, cross-statistic, and classifier terms. The
quadratic Gram/factor terms remain asymptotically quadratic in width.
