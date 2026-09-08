# SRQ generalization M6 width-sweep runbook

Status: implementation ready; the real CIFAR-100 train-only GPU gate has not
yet been run. M6 does not authorize held-out test evaluation or width
selection.

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
