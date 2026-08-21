# SRQ-FLY results ledger

Every manuscript number must map to an immutable artifact or audited research
record. Do not copy metrics from notebook display text without recording its
artifact hash and status here.

## ImageNet-R D2

- status: `PASS_REVIEW_D2`, train-validation only;
- evidence ZIP SHA-256:
  `e0c2ef9f94b21cfcedadd9a0f7dbe05e9abae4e86b88593d42a90411d5afb033`;
- result SHA-256:
  `a7f08b4608e9f571da7698bab35bf64d967a18f01b9b2e4818ad1ee99a535263`;
- source:
  `docs/research/SRQ_FLY_D2_STATE_MATCH_IMPLEMENTATION.md`;
- SRQ-FLY-10000: AA `77.9343`, final `71.1197`, state `105166628` B;
- exact FLY-4518: AA `77.0141`, final `70.0085`, state `105149848` B;
- paired difference: AA `+0.9201`, final `+1.1112` points.

## ImageNet-R D2.1

- status: `PASS_REVIEW_D21`, nested train-validation only;
- evidence ZIP bytes: `9949`;
- evidence ZIP SHA-256:
  `c41f681a8b6b69e032b9df765f801888707761dd3ee6354d860df7669b88ddae`;
- lambda-selection SHA-256:
  `dd8b2b3f6d2c68c20d7e1038511c1f690e947c987933d8af73790303609d1920`;
- source: `docs/research/SRQ_FLY_D21_LAMBDA_ROBUSTNESS_PROTOCOL.md` and the
  immutable user-supplied evidence ZIP;
- exact FLY-4518 selected `lambda=1e6` from
  `1e4,1e5,1e6,1e7,1e8,1e9` using inner training-validation only;
- outer exact FLY-4518: AA `77.0141`, final `70.0085`, state `105149848` B;
- all locked D2.1 gates passed and held-out test authorization remained false.

## CUB D3

- status: formal `STOP_SRQ_FLY_D3`, train-validation only;
- evidence ZIP SHA-256:
  `4d2104c80e3f5fa125839f7723ac86126fbb0395c53b7307a9de1a349b8f380a`;
- result SHA-256:
  `f172d508c14fd95e7dcece5cd22c04a8e9f88c35c638fa140b476ebf6d4e6f4b`;
- source: `docs/research/SRQ_FLY_D3_CUB_RESULTS.md`;
- failure: rejected inner candidates exceeded the historical `1e-5`
  numerical gate; selected and outer candidates remained below it;
- SRQ-FLY-10000: AA `91.7564`, final `87.5282`, state `105166628` B;
- exact FLY-10000: AA `91.6761`, final `87.1102`, state `452006940` B;
- exact FLY-4518: AA `91.5147`, final `86.9393`, state `105149848` B;
- raw Ridge: AA `89.6107`, final `84.8446`, state `7177792` B.

## CUB D4

- status: formal `STOP_SRQ_FLY_D4`, train-validation only;
- evidence ZIP SHA-256:
  `6b65500e1d1dccb631f02d5c2016e55451f47762518a175dde7a40d6431c7fa1`;
- result SHA-256:
  `713d7a31fa09a6f983ffd090bbb40c1be5c9e69a8585717cf3bc9a86451323ac`;
- source: `docs/research/SRQ_FLY_D4_CUB_RESULTS.md`;
- failure: seed 2027 task 20 agreement `97.8178%` versus the locked `98%`
  minimum; all other gates passed;
- exact FLY-10000 mean: AA `91.6070`, final `87.2753`;
- SRQ-FLY-10000 mean: AA `91.5699`, final `87.1086`;
- exact FLY-4518 mean: AA `91.0547`, final `86.5035`;
- raw Ridge mean: AA `90.0239`, final `83.9767`;
- SRQ minus FLY-4518: mean AA `+0.5153`, mean final `+0.6051`;
- five-seed AA-gain sample standard deviation: `0.5086`;
- five-seed AA-gain 95% t interval: `[-0.1162, +1.1467]`;
- SRQ wins: four of five seeds;
- SRQ state: approximately 23.27% of exact FLY-10000 and within 0.1% of
  exact FLY-4518 for every seed.

## Missing paper evidence

- no CUB held-out result;
- no ImageNet-R held-out result; the current processed artifact failed the
  content-disjointness audit with 19 cross-split duplicate hashes;
- no second-backbone result;
- no controlled paper-hardware runtime comparison;
- no error-feedback or lower-bit result;
- the bibliography has a verified primary-source ledger but still needs an
  exhaustive venue-specific literature review.
