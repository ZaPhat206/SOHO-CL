# SRQ-FLY D4 CUB Colab runbook

Use `notebooks/srq_fly_cub_d4_multiseed_colab.ipynb` on a Colab T4 GPU.

1. Push `feature/srq-fly-d4-multiseed` before running the notebook.
2. Edit path/source values in cell 2 only; never edit seeds, lambda, grid,
   split, state identities, or gates.
3. Run cells in order. The notebook restores the existing verified CUB
   train-only feature cache and D3 evidence from Drive.
4. The runner prints 40 short raw-search START/DONE lines, then one
   `SEED START/DONE` block per seed with WTA and task progress.
5. Five fresh seeds can take materially longer than D3, but completed units and
   WTA caches resume after disconnection. Do not restart with a changed output
   directory unless a stale-context error is diagnosed.
6. Download `srq_fly_cub_d4_multiseed_train_only.zip` from the final cell and
   return it for audit.
7. Stop. Never create CUB `test.pt`, even if D4 prints `PASS_REVIEW_D4`.

The ZIP contains result/config/log evidence only. Drive feature/WTA caches are
sample-level experiment infrastructure, not exemplar-free learner state.
