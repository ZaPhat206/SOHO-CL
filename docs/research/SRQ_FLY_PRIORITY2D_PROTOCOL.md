# SRQ-FLY Priority 2D: real train-only backend equivalence

Priority 2C passed its synthetic CUDA gate.  Priority 2D is the final pre-test
integration check: it compares the selected implicit-Ridge backend against the
Priority-2B backend on frozen CIFAR-100 **training features only**.

Both methods use the same seed 2025, ten-task class order, deterministic 80/20
per-class train-validation split, width 10,000 projection, WTA code cache,
Ridge value, int8 factor format, and quantization batch 64.  The only difference
is the first-update path.  Hyperparameters are inherited; this phase performs
no selection.

The runner uses isolated processes and accepts the candidate only when:

- every stage accuracy differs by at most `0.01` percentage points;
- relative fixed-probe logit drift is at most `1e-5`;
- persistent state bytes are identical;
- solver residual is at most `2e-5`;
- peak allocated CUDA memory does not exceed Priority 2B;
- total update time does not exceed 1.25 times Priority 2B.

The feature and WTA caches are experiment infrastructure and are not learner
state.  `test.pt` must be absent throughout.  `PASS_FINAL_BACKEND` locks the
implementation for the separately authorized three-dataset held-out run; it
does not itself authorize or consume held-out data.
