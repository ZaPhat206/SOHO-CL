# SRQ-FLY D2.1 nested lambda-robustness protocol

Status: locked train-only falsification study. It does not authorize held-out
ImageNet-R evaluation.

## Question

D2 found that SRQ-FLY-10000 exceeded exact FLY-4518 by 0.9201 percentage point
in validation average accuracy and 1.1112 points at task 20 while their
persistent tensor states differed by only 0.01596%. Both used the D1 constant
`lambda=1e6`. D2.1 asks whether a dimension-specific Ridge coefficient removes
that state-matched advantage.

## Nested selection

The D1 split remains the outer protocol: 80% training and 20% outer validation
within every class and task. D2.1 deterministically subdivides only the outer
training indices into inner-fit and inner-validation partitions, again using
seed 2025 and a 20% inner-validation fraction.

Candidate lambdas are locked before execution:

```text
1e4, 1e5, 1e6, 1e7, 1e8, 1e9
```

This range contains the previously locked `1e6`, the lower `1e4` control used
by the existing ImageNet-R FLY studies, and the original FLY GCV exponent range
through `1e9`. It is not selected from D2 outer-validation performance.

For each candidate, exact FLY-4518 is trained on the cumulative inner-fit
stream and scored on the seen inner-validation partitions after every task.
The candidate with maximum inner average incremental accuracy is selected;
an exact tie chooses the smaller lambda. Candidate evaluators receive only the
inner-fit and inner-validation index partitions; the outer-validation indices
are used only by the later locked evaluation. No sample index is serialized.

After `lambda_selection.json` is written, exact FLY-4518 is refit on all outer
training indices and evaluated once on the outer-validation stream. SRQ metrics
are immutable values imported from the verified D2 artifact; SRQ is not rerun
or retuned.

## Locked controls and state

Backbone, preprocessing, feature cache, projection rows, expansion dimension,
Top-K ratio, class order, task split, seed, dtype, and state accounting are
identical to D2. The WTA cache is sample-level experiment infrastructure and is
not learner state or part of the evidence ZIP.

Exact FLY persistent state contains only the fixed sparse projection, Gram,
cross-statistic, counts, and derived classifier. It stores no historical image,
feature, code, label, or sample index.

## Decision

D2.1 passes for review only when:

- the selected value belongs to the locked candidate list;
- every inner candidate and the single outer evaluation complete with maximum
  solve residual at most `1e-5`;
- measured exact-FLY state equals 105,149,848 bytes and remains within 0.1% of
  the D2 SRQ state;
- SRQ exceeds tuned exact FLY-4518 by at least 0.10 percentage point in outer
  validation average accuracy;
- SRQ is not worse in task-20 outer-validation accuracy;
- `test.pt` remains absent and the supplied D2 artifact is an exact clean,
  train-only match.

A pass supports the SRQ state-efficiency interpretation under a
dimension-specific lambda and permits design of an unseen-dataset train-only
replication. A failure closes the current accuracy-at-matched-state claim. It
does not authorize changing this grid, seed, split, or opening held-out data.
