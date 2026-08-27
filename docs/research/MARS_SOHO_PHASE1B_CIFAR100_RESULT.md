# MARS-SOHO Phase 1B CIFAR-100 result

Artifact `mars_soho_phase1b_cifar100_train_only.zip` has SHA-256
`9f9844683c4a823605fd34d801e63e0ebe71f7fd85af5dc69a8ad52273b1b87d`.
It was produced from clean commit `02a4028581d1be32c802846ca48c0357c743f65b`
on a Tesla T4 with Python 3.13.15 and torch 2.11.0+cu128. All embedded source
hashes match the provenance record, and `uses_test_set` is false.

## Decision

Phase 1B failed its scientific gate. It repaired the Phase-1 allocation
collapse, but allocation was not the dominant error source.

| Method | AIA mean +/- SD | Final mean +/- SD |
|---|---:|---:|
| exact replay oracle | 90.671 +/- 0.180 | 85.650 +/- 0.221 |
| statistic-variance aware | 88.838 +/- 0.023 | 83.443 +/- 0.246 |
| shuffled statistic variance | 88.833 +/- 0.044 | 83.490 +/- 0.297 |
| heterogeneous uniform | 88.811 +/- 0.029 | 83.413 +/- 0.278 |
| shuffled turnover | 88.810 +/- 0.021 | 83.400 +/- 0.229 |
| turnover aware | 88.807 +/- 0.030 | 83.420 +/- 0.221 |

Statistic-variance allocation improved AIA over uniform by only `0.027` pp
and over its shuffled control by `0.004` pp. Paired 95% intervals were
`[-0.096,0.149]` and `[-0.088,0.097]` pp respectively. Its gap to exact replay
was `1.834` pp with paired interval `[-2.292,-1.375]` for proposed minus
oracle.

All 27 eligible stages had noncollapsed risk, nonuniform allocation and an
allocation distinct from the shuffled control. Statistic-variance allocation
ranged from 56 to 70 pseudo-directions per class; turnover allocation ranged
from 54 to 74. The old sufficient certificate remained saturated at one.

The exemplar-free state remained 13,208,464 bytes with zero historical rows,
90.32% below the matched replay-oracle state. Persistent bytes are not peak
runtime memory.

## Interpretation

The experiment separates implementation failure from hypothesis failure:
continuous risk and Neyman allocation worked as designed, yet did not close
the oracle gap. The remaining error is therefore dominated by distributional
reconstruction bias rather than fixed-budget Monte-Carlo allocation variance.
Do not open test data, extend to other datasets or add SRQ. Phase 1C must audit
and improve the reconstructed distribution itself.
