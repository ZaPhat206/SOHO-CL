# SRQ-FLY CIFAR D5 Colab runbook

Use `notebooks/srq_fly_cifar_d5_train_only_colab.ipynb` on a Colab T4 GPU.

1. Ensure the branch in cell 2 contains the notebook, runner and locked config.
2. Set `DRIVE_FEATURE_CACHE` to the existing CIFAR frozen-feature cache. The
   notebook copies only `metadata.json` and `train.pt`; it never copies or
   opens `test.pt`.
3. Run cells top to bottom. Do not change seed, grid, representation, state
   match, Ridge controls or gates.
4. Cell 5 runs synthetic correctness tests.
5. Cell 6 prints bounded progress:
   - `WTA CACHE`: one-time/resumable code encoding;
   - `INNER START/DONE` and `TASK`: five lambda candidates;
   - `OUTER START/DONE`: locked paired controls.
6. Cell 7 downloads `srq_fly_cifar_d5_train_only.zip`. Return that ZIP for
   audit and stop.

The two WTA caches remain on Drive so an interrupted run can resume. They may
occupy roughly 1.3 GB combined and are experiment caches containing
sample-level codes, not exemplar-free learner checkpoints.
