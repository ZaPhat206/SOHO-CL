# SRQ generalization M12: locked test confirmation

Status: complete. The locked run produced all 36 units and the final artifact
`srq_generalization_m12_locked_test_confirmation.zip` (SHA-256
`02ada7180e66e780be77c7934b87d268f0d171dfee1c318fd3e9ae7e48846b1e`).
This protocol froze the final RanPAC confirmation before test features were
materialized.

## Purpose and scope

M12 confirms the train-only method decisions from M6, M11, and M11b on the
official CIFAR-100 test split. It does not perform another method search.
The test split was already consumed by earlier FLY experiments, so this is a
source-locked confirmation of the RanPAC transfer rather than a fresh
first-use held-out benchmark.

The evaluated frontend remains the controlled RanPAC Phase-2 random-ReLU head:
frozen ViT-B/16 features, no PETL reproduction, standard-normal projection,
and a global additive-Ridge classifier. Consequently, M12 cannot be described
as reproduction of published end-to-end RanPAC accuracy.

## Frozen choices

- widths: 10,000 and 20,000;
- Ridge coefficient: $10^6$ at both widths, inherited from M6;
- methods: Exact Gram, fixed P2B INT8/FP32, and M11 adaptive INT8/FP16;
- P2B storage: block 256, group 64, QR panel 128, streaming batches of 64
  blocks;
- adaptive policy: the single M11 25% byte-interval rule;
- six paired class-order/projection replicates: `3031/5031` through
  `3036/5036`;
- full official training split for fitting and official test examples from
  seen classes after each task for evaluation.

M11b refined INT8 is explicitly excluded because it formally failed its
locked train-only width-20,000 gate. Its test accuracy must not be inspected.
No alternative width, Ridge value, precision allowance, method, seed, or retry
may be selected after M12 begins.

## Protocol-recovery disclosure

An initial M12 execution materialized test metrics but terminated before a
final artifact was written. The failure came from an invalid state-byte gate:
it required the value-sensitive adaptive mask under every locked projection
seed to reproduce the exact intermediate bytes of the single M11 development
seed. The recovery changes only this non-accuracy integrity check to the
M11-locked all-INT8 floor and 25% factor-budget ceiling. No method, width,
Ridge value, seed, precision policy, or accuracy decision changed after the
test output was observed. This engineering recovery is not an additional
method-selection retry and must remain disclosed with the final result.

## Authorization boundary

The `authorize` command must run from a clean committed checkout while only
`train.pt` is visible. It verifies the exact M6, M11, and M11b ZIP identities,
their embedded result hashes and statuses, the train cache, configuration,
current commit, and frozen method choices. It then writes a deterministic
authorization record.

Only the `extract-test` command accepts that authorization and writes
`test.pt`. It extracts test features with the same checkpoint and preprocessing
as the authorized training cache and binds the cache metadata to the
authorization ID. The `run` command refuses a cache or source artifact that
does not match this identity.

Two environment-specific notebooks implement the same contract. The Colab
notebook stores resumable units in Google Drive. The Kaggle notebook pins the
complete source commit, discovers source artifacts and CIFAR from the
read-only `/kaggle/input` mounts, keeps the sample-level feature cache in
non-persistent `/kaggle/temp`, and writes only authorization, unit summaries,
plots, and the final ZIP beneath `/kaggle/working`. Environment changes do not
change any experimental choice or gate.

Kaggle Datasets extract recognized archives server-side. Therefore, each of
the three source ZIPs must be renamed locally by appending `.bin` before it is
uploaded (for example, `artifact.zip` becomes `artifact.zip.bin`), without
extracting or recompressing it. The Kaggle notebook accepts either an exact ZIP
that remains available or this canonical `.zip.bin` form, verifies the frozen
outer SHA-256, and copies the unchanged bytes to an exact `.zip` filename under
`/kaggle/temp` before authorization. Reconstructing a ZIP from Kaggle-extracted
members is intentionally forbidden because archive metadata would change the
locked outer identity.

## Reporting contract

M12 reports every replicate, task, width, and method. Primary metrics are AIA
and final accuracy. Tables report mean and sample standard deviation, plus
paired method-minus-Exact differences. Persistent tensor bytes, analytic
update time, representation time, and solver residual are also retained.

There is deliberately **no accuracy gate**. Completion depends only on source
identity, prior authorization, exact sample/class inventory, all 36 units
being present, finite metrics, and solver residual at most `2e-5`. Exact Gram
and fixed P2B INT8 must retain task-wise byte identity with M6. Adaptive
INT8/FP16 is value-sensitive: different locked projection seeds may choose
different mixtures of unequal-size edge blocks, especially when the width is
not divisible by 256. It is therefore checked task-wise against the locked M11
all-INT8 floor and 25% factor-budget ceiling, while the policy, block layout,
mask dtype, and accounting identity remain fixed. An unfavorable accuracy
result is still a completed result and cannot trigger a retry.

## Completed result

The authorization ID is
`c5599009d760cdfc85a1846ba4c762428afa39a739a3fc73efdc8a34ce5abe99`.
The final result has status `COMPLETE_M12_LOCKED_TEST_CONFIRMATION`; its
`m12_results.json` SHA-256 is
`e6af9e735fe64a8a40177a4dbc2cb38979ccae6a51ccc35e53d5654c08ee1f25`.
All 36 units completed, all source/state/budget/inventory gates passed, and the
maximum solver relative residual was `7.243e-6`, below the locked `2e-5`
threshold. Accuracy remained descriptive and was not a completion gate.

Values below are mean ± sample standard deviation over six paired replicates.
Differences are method minus Exact at the same width.

| Width | Method | Test AIA | Final | AIA difference | Final difference | State | State reduction | Update/Exact |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | Exact | 92.5783 ± 0.3829 | 89.128 ± 0.075 | -- | -- | 418.40 MiB | -- | 1.00× |
| 10,000 | P2B INT8/FP32 | 92.4444 ± 0.4135 | 88.905 ± 0.135 | -0.1339 pp | -0.223 pp | 87.62 MiB | 79.06% | 1.88× |
| 10,000 | Adaptive INT8/FP16 | 92.5731 ± 0.3886 | 89.115 ± 0.067 | -0.0052 pp | -0.013 pp | 98.80 MiB | 76.39% | 2.79× |
| 20,000 | Exact | 92.8220 ± 0.4147 | 89.660 ± 0.113 | -- | -- | 1,599.73 MiB | -- | 1.00× |
| 20,000 | P2B INT8/FP32 | 92.4956 ± 0.4860 | 89.003 ± 0.150 | -0.3264 pp | -0.657 pp | 276.57 MiB | 82.71% | 1.91× |
| 20,000 | Adaptive INT8/FP16 | 92.8266 ± 0.4111 | 89.678 ± 0.087 | +0.0046 pp | +0.018 pp | 321.28 MiB | 79.92% | 2.91× |

The locked multi-seed result confirms the train-only diagnosis: fixed P2B is
the smallest tested state but loses more accuracy at width 20,000. Adaptive
SRQ tracks Exact closely at both widths while spending more state and update
time than fixed P2B. Its small positive mean differences at width 20,000 are
not evidence that quantization improves accuracy.

## Permitted conclusion

Completion permits a multi-seed test-confirmation statement for the specified
RanPAC/random-ReLU additive-Ridge setting. It does not establish a fresh
untouched benchmark, official RanPAC reproduction, official GACL
reproduction, compatibility with changing historical representations, or a
universal plug-in claim.
