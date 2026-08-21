# SRQ-FLY D0 ImageNet-R Colab runbook

Use `notebooks/srq_fly_imagenetr_d0_colab.ipynb` on a Colab T4 GPU. This is a
locked five-task train-validation diagnostic. It is not a held-out evaluation.

## Purpose

The notebook tests whether SPD-by-construction square-root quantization is a
viable replacement for FLY's exact 10,000-dimensional Gram matrix. It compares:

- exact FLY at dimensions 10,000 and 4,096;
- raw-feature Ridge;
- direct groupwise-int8 Gram storage;
- float16 square-root storage;
- groupwise-int8 square-root storage (SRQ-FLY).

The compact exact-FLY control is mandatory: SRQ-FLY is not useful if a smaller
ordinary FLY representation has equal-or-better accuracy with no more state.

## How to run

1. Push branch `feature/srq-fly-d0` before opening Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Open `notebooks/srq_fly_imagenetr_d0_colab.ipynb`.
4. Edit only repository/checkpoint/Drive paths in cell 2 if necessary.
5. Run cells 1 through 10 in order.
6. Follow `WTA CACHE`, `START`, `TASK`, `DONE`, and `RESUME` lines. A completed
   unit is saved under Drive and safely resumes after interruption.
7. Download `srq_fly_imagenetr_d0_train_only.zip` from cell 10 and return it for
   audit. Stop; do not create or evaluate `test.pt`.

The existing 10k WTA cache is reused from
`tail_fly_imagenetr_wta_cache_seed2025`. A new 4,096-dimensional cache is saved
separately. Both are sample-level experiment infrastructure, not learner state,
and must never be included in an exemplar-free checkpoint.

## Locked identity

- config: `configs/srq_fly_imagenetr_d0_train_only.json`;
- config SHA-256: `039e243543c46d8f4ab6984197feccbc4ac4e9d8f96dd6e80f19c49e6460fbd1`;
- seed: `2025`;
- diagnostic scope: first `5` of `20` class-incremental tasks;
- fixed FLY Ridge: `1e6` (a diagnostic constant inherited from the prior
  train-only exact-FLY schedule, not tuned in D0);
- held-out test use: prohibited.
