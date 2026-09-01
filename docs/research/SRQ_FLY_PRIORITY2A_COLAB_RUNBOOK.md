# SRQ-FLY Priority 2A Colab runbook

Use `notebooks/srq_fly_priority2a_colab.ipynb` with a Tesla T4 runtime.

1. Edit only repository URL/branch and output paths in cell 2.
2. Run cells in order.
3. The correctness cell uses synthetic CPU tests.
4. The benchmark prints one `START`, two `TASK`, and one `DONE` line for every
   method/repetition.  One warm-up plus seven measured rounds are expected.
5. Do not change the chunk grid, gates, task dimensions, seed, panel size, or
   repetitions after seeing output.
6. Download `srq_fly_priority2a_memory.zip` and return it for audit.

No dataset or checkpoint is downloaded.  The notebook never creates or opens
`train.pt` or `test.pt`.  `PASS_REVIEW_PRIORITY2A` selects a system candidate
for later train-only review; it does not authorize held-out evaluation.
