# SRQ generalization M12: locked test confirmation

Status: implementation complete; locked test run pending. This protocol
freezes the final RanPAC confirmation before test features are materialized.

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

## Permitted conclusion

Completion permits a multi-seed test-confirmation statement for the specified
RanPAC/random-ReLU additive-Ridge setting. It does not establish a fresh
untouched benchmark, official RanPAC reproduction, official GACL
reproduction, compatibility with changing historical representations, or a
universal plug-in claim.
