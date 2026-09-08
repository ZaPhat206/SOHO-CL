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

## M0 paper evidence freeze

The machine-readable source of truth for the current manuscript tables is
`paper/EXPERIMENTS_MANIFEST.json`. It distinguishes fresh train-only evidence,
test-used confirmation, and recovery evidence. The current paper uses:

- P2B same-width confirmation, artifact SHA-256
  `14826488b8d82bc306a07e6d4f229cc389a8447150833aefc1de664961a9e85d`;
- state-matched secondary control, artifact SHA-256
  `a5adc883089f6108a01f33d57f0737894af843262a18a50f5309d82a54f323f9`;
- direct-quantization train-only control, artifact SHA-256
  `9c5f8c9c0d945393cea48d23204ed585e42e656359bbb12386691f4ba452988e`;
- task-frequency train-only control, artifact SHA-256
  `2891b3ca7ed53c62bd63aa1cb5b3dabb374f4de392de6fa5ccf14fb7bb6690c4`;
- whole-process train-only memory audit, artifact SHA-256
  `7f111e80ec3e4d12fafae39a868795fc36c967d223c99f8ade98107b5b180403`.

The P2B and state-matched results use previously consumed test splits and are
not fresh first-use held-out evidence. The current state-matched archive also
retains its disclosed runtime-adapter caveat and is not promoted to primary
source-locked evidence by this ledger update.

## Open paper evidence

- no independent non-FLY analytic frontend has passed a plug-in gate;
- no low-rank or streaming-sketch baseline has been evaluated at matched
  persistent bytes;
- no second-backbone result;
- whole-process memory has one isolated run per method on one Tesla T4 rather
  than a repeated or cross-hardware interval;
- the legacy ImageNet-R processed split failed the content-disjointness audit
  with 19 cross-split duplicate hashes;
- no error-feedback or true packed lower-bit result;
- the bibliography has a verified primary-source ledger but still needs an
  exhaustive venue-specific literature review.

## Generic backend M3 FLY regression

- status: `PASS_M3_FLY_REGRESSION`, CIFAR train-only;
- evidence ZIP SHA-256:
  `ec376875ff0e29e2d7a46d1fcb9fcccc523ea47fbe6a8f2dd83ec3a98930a595`;
- result SHA-256:
  `bc2851c75362ac4bda8cd0ab022d21c4f952b1310dafbabaa7cf74452b98514c`;
- source commit: `56d525f453744f13ff287507f6ec4c30aaf81644`, clean checkout;
- all ten task records have exact legacy/generic tensor and persistent-byte
  identity for both Exact FLY and P2B;
- minimum prediction agreement: `100%`;
- maximum relative logit error: `0`;
- maximum solver relative residual: `3.132620984020085e-6`;
- final Exact/P2B state: `444006540 / 97166228` B;
- P2B state reduction: `78.1160%`.

At M3 alone, this result validated backend extraction for FLY but did not yet
close the non-FLY frontend evidence item. M4 below supplies that next gate.

## Generic backend M4 RanPAC train-only gate

- status: `PASS_M4_RANPAC_TRAIN_ONLY`, CIFAR train-only;
- evidence ZIP SHA-256:
  `228da0828c7f6964bcc8f7da92258efa20b00eef0fc83679246ce0ae2a0c8f52`;
- result SHA-256:
  `1afb59eeb6047c1dcdde2da14da27c0c8763b1b06bf43baa8f9b38019e39cfcc`;
- source commit: `47a789b6656e88d2ae5a3211f34eea485636e10a`, clean checkout;
- scope: controlled RanPAC Phase-2 standard-normal random projection and ReLU
  analytic head; no PETL and no reproduction of the original per-task Ridge
  schedule;
- fixed Ridge selected from train-only MSE calibration: `lambda=1e6`;
- validation AIA, Exact / FP32 square-root / FP16 square-root / P2B:
  `92.6222 / 92.6222 / 92.6215 / 92.4608`;
- final validation accuracy: `88.50 / 88.50 / 88.53 / 88.39`;
- Exact-minus-P2B validation-AIA gap: `0.1614` percentage points;
- final total state, Exact / FP16 / P2B:
  `438720400 / 138750400 / 91880088` B;
- final quadratic state, Exact / FP16 / P2B:
  `400000000 / 100030000 / 53159688` B;
- P2B reduction: `86.7101%` of quadratic state and `79.0573%` of total state;
- dense FP32 square-root minimum prediction agreement: `100%`;
- maximum FP32 system/weight/logit relative errors:
  `2.10e-6 / 1.24e-5 / 9.43e-5`;
- maximum solver relative residual: `4.65e-6`;
- all eight locked gates passed and no numerical failure occurred.

This closes the initial non-FLY frontend evidence item. It supports "a
reusable backend demonstrated on two analytic learners," not a universal
plug-in claim. The result is one development seed and does not constitute a
published-accuracy reproduction of full RanPAC.
