# SRQ-FLY D4: five-fresh-seed CUB train-only confirmation

Status: prospective preregistered protocol. It does not authorize CUB held-out
evaluation.

## Purpose

D3 seed 2025 supported every state/accuracy hypothesis but formally stopped
because two rejected inner candidates exceeded a universal `1e-5` residual
gate by at most `1.40e-6`. D4 does not relabel or rerun D3. It asks whether the
state-efficiency ordering replicates over five fresh seeds:
`2026, 2027, 2028, 2029, 2030`.

The four paired methods are exact FLY-10000, SRQ-FLY-10000 int8, exact
FLY-4518 at matched state, and float64 streaming raw Ridge. All methods use the
same frozen ViT cache, task order, outer split, and inference examples within a
seed. No learner receives Task-ID or retains images/per-sample features.

## Fast prospective design

D4 transfers the D3-selected FLY configuration unchanged:

- expansion 10,000 versus matched width 4,518;
- synaptic degree 300 and Top-K ratio 0.3;
- block/group sizes 256/64;
- FLY Ridge `lambda=1e5`.

There is no repeated 10K FLY search in D4. This both avoids tuning on the new
outer folds and removes 60 expensive Cholesky streams per seed. Each fresh seed
therefore runs exactly one paired exact/SRQ-10000 stream and one exact-FLY-4518
stream.

Raw Ridge was at the D3 grid boundary, so D4 selects one **global** raw lambda
by mean inner-validation accuracy across all five seeds from
`1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000`. That selected value is then used
once on every untouched outer fold. A boundary selection fails the bracketing
gate but does not trigger an after-the-fact grid extension.

## Numerical rule

The rule is declared before any D4 score is observed:

- discarded raw-search candidates must be finite with relative solve residual
  at most `2e-5`;
- every reported outer model must have relative solve residual at most `1e-5`.

The two thresholds separate search validity from paper-facing model validity.
They apply only prospectively; D3 remains a formal STOP.

## Primary gates

D4 passes review only if integrity/state/numerical checks pass and:

- SRQ is within 0.50 point of exact FLY-10000 in average and final accuracy
  for every seed;
- minimum paired prediction agreement is at least 98% for every seed;
- SRQ uses at most 25% of exact FLY-10000 state and matches FLY-4518 state
  within 0.1% for every seed;
- mean and median SRQ average gain over FLY-4518 are each at least 0.10 point;
- mean final gain over FLY-4518 is non-negative;
- SRQ wins average accuracy on at least four of five seeds;
- no seed loses more than 0.25 point to FLY-4518;
- raw Ridge does not Pareto-dominate SRQ on mean accuracy/state.

Mean, sample standard deviation, individual paired values, and a two-sided 95%
t interval are reported even when a gate fails. A pass authorizes review of a
separately preregistered held-out protocol, never automatic test access.

Locked config: `configs/srq_fly_cub_d4_multiseed_train_only.json`, SHA-256
`a0742a545fa83f54b18bcf2372ea6d6e518d214f48f882f2b94696122c2fe8fd`.

## Cache/state separation

The CUB train feature cache and ten per-seed WTA caches are sample-level
experiment infrastructure on disk. They are excluded from the evidence ZIP
and learner checkpoint. Persistent state is measured independently after each
complete stream and contains only projection/statistics/classifier tensors.
