# SRQ-FLY D2 state-matched falsification protocol

Status: locked train-only control study. D2 is an audit required by the mixed
D1 result; it is not a continuation authorized by calling D1 a pass, and it
does not authorize held-out evaluation.

## Question

D1 showed that SRQ-FLY retained exact FLY-10000 accuracy while reducing final
persistent tensor state from 452,006,940 to 105,166,628 bytes. Exact FLY-4096
used only 88,415,016 bytes, so it was not an exact state match. D2 asks whether
ordinary dimension reduction at the same state budget explains SRQ-FLY's gain.

Under the repository tensor accounting, exact FLY with feature dimension `m`,
input dimension 768, synaptic degree 300, and 200 classes stores

\[
S_{exact}(m)=4m^2+5200m+6952\text{ bytes}.
\]

The closest integer dimension not exceeding the observed SRQ budget is
`m=4518`, for which `S_exact=105,149,848` bytes: only 16,780 bytes (0.01596%)
below SRQ. Dimension 4519 would exceed SRQ by 24,568 bytes. The dimension is
therefore selected from state accounting alone, not validation accuracy.

## Locked experiment

D2 reuses D1's train feature cache, seed 2025, class order, 20% validation
split, fixed Ridge `lambda=1e6`, projection distribution, synaptic degree, and
coding level. It creates one new WTA experiment cache for `m=4518` and evaluates
only exact FLY-4518 over all 20 tasks.

With the unchanged row-wise FLY projection generator and seed 2025, its 4518
projection rows are exactly the first 4518 rows of the FLY-10000 projection.
Thus the control changes the expansion budget without introducing a different
random projection realization. A synthetic exact-equality test locks this
property.

The runner requires the exact D1 `d1_results.json` and verifies its commit,
config, train artifact, split hashes, SRQ state, average accuracy, final
accuracy, and absence of held-out use before making the comparison. No
hyperparameter is selected in D2.

## Decision

D2 supports the SRQ state-efficiency claim only if:

- exact FLY-4518 completes with residual at most `1e-5`;
- its measured state matches the analytic 105,149,848-byte prediction and is
  within 0.1% of the D1 SRQ state;
- SRQ exceeds exact FLY-4518 by at least 0.10 percentage point in validation AA;
- SRQ is not worse in final seen-class validation accuracy;
- `test.pt` remains absent.

If exact FLY-4518 matches or exceeds SRQ, dimension reduction is the simpler
explanation and the current SRQ state-efficiency claim is closed. A D2 pass
supports SRQ only under the locked D1 `lambda=1e6`; it does not prove that SRQ
dominates a dimension-specific, independently selected FLY-4518 configuration.
It therefore permits design of a separately locked train-only lambda-robustness
control before any unseen-dataset replication, not a test-set run.
