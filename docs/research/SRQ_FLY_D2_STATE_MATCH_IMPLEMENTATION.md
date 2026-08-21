# SRQ-FLY D2 state-match implementation record

Status: exact state-matched control implemented and synthetically verified;
real ImageNet-R D2 has not been run.

## Scope

D2 adds no learner and changes no SOHO/FLY implementation. It derives exact
FLY dimension 4518 solely from the observed 105,166,628-byte D1 SRQ state,
creates the corresponding fixed WTA cache, and evaluates exact FLY-4518 on the
locked 20-task training-validation stream.

The runner binds itself to D1's commit, config, train artifact, task/split
hashes, and SRQ metrics. It refuses a visible `test.pt`, stale D1 result,
different split, non-state-matched dimension, or inconsistent analytic/runtime
state accounting.

## Verification

```text
python -m pytest -q tests/test_srq_fly_math.py tests/test_srq_fly_d2_state_match.py
12 passed, 20 warnings in 6.00s

python -m pytest -q
235 passed, 20 warnings in 16.54s

python -m json.tool configs\srq_fly_imagenetr_d2_state_match.json
PASS

python -m json.tool notebooks\srq_fly_imagenetr_d2_state_match_colab.ipynb
PASS

git diff --check
PASS
```

The sparse-CSC warnings come from the unchanged FLY projection. Synthetic tests
verify the 105,149,848-byte analytic result for FLY-4518, runner resume,
held-out refusal, D1 identity enforcement, rejection of duplicate JSON keys,
rejection of a non-closest dimension, and exact seed-matched projection-prefix
construction.

The supplied D1 ZIP was also audited read-only before handoff. Its SHA-256 is
`1fc26111c885bdf8c2c5056f360d6eef2acbe368da23c6e9bce8b4998f05d63c`.
The embedded config hash, clean runner commit, train/split hashes, SRQ state and
metrics all match D2's locked reference; it records no held-out use. The final
D2 config SHA-256 is
`e8c630b728f9b5f554fd94e6d450b3db4b2205d0d94a595095fa7ebdddcda197`
and is bound identically in the Colab notebook and runbook.

## Handoff

After commit and push of `feature/srq-fly-d2-state-match`, run cells 1 through 7
of `notebooks/srq_fly_imagenetr_d2_state_match_colab.ipynb`. Return
`srq_fly_imagenetr_d2_state_match.zip` and stop without held-out evaluation.
