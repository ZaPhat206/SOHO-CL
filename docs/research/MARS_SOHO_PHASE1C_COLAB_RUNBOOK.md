# MARS-SOHO Phase 1C Colab runbook

Use `notebooks/mars_soho_phase1c_fidelity_colab.ipynb` on a Colab T4. This is a
CIFAR-100 train-only reconstruction-fidelity study.

1. Run cells from top to bottom without editing the locked config, seeds, rank
   grid, pseudo budget or gates.
2. The notebook restores a runtime train-feature cache if available; otherwise
   it extracts CIFAR-100 training features only with progress.
3. Correctness tests run before the fidelity study.
4. Rank selection prints `START/RESTORED/DONE` for nine inner units. Outer
   fidelity then runs twelve paired method/replicate units.
5. Download `mars_soho_phase1c_cifar100_train_only.zip` and return it for audit.
6. Stop. Do not expose test features or integrate the candidate into the
   continual learner before review.

The exported ZIP excludes frozen per-sample feature caches. Those caches are
experiment infrastructure and never learner state.
