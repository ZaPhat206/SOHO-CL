# MARS-SOHO Phase 1B Colab runbook

Use `notebooks/mars_soho_phase1b_train_only_colab.ipynb` on a Colab T4. Run
only CIFAR-100 first. This is a train-only validation study, not a held-out test
run.

1. Select **Runtime -> Change runtime type -> T4 GPU**.
2. Keep `DATASET_KEY='cifar100'` and run cells from top to bottom.
3. Do not edit the locked config, seeds, allocation rules or gates.
4. The notebook may reuse a local runtime train-feature cache; the learner
   checkpoint never includes that cache.
5. Wait for six methods times three paired replicates. Progress prints one
   `TASK` line per continual stage and each completed unit is resumable while
   the runtime disk remains available.
6. Download `mars_soho_phase1b_cifar100_train_only.zip` and return it for audit.
7. Stop. Do not expose `test.pt`, start CUB/ImageNet-R, or add SRQ before the
   CIFAR-100 Phase-1B gate is reviewed.

The result must be described as train-only evidence. A noncollapsed allocation
does not by itself establish improved accuracy; all preregistered gates must
pass.
