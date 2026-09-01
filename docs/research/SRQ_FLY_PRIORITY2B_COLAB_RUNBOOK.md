# SRQ-FLY Priority 2B Colab runbook

Use `notebooks/srq_fly_priority2b_colab.ipynb` with a Tesla T4 runtime.

1. Edit only repository URL/branch and output paths in cell 2.
2. Run cells in order.
3. The correctness cell is CPU/synthetic only.
4. The CUDA cell runs one warm-up plus seven measured rounds.  Each round has
   Exact FLY, eager SRQ, and four streaming batch candidates.
5. Completed worker JSON/probe pairs resume only when config, source hashes,
   method identity, and profiling mode match.
6. Do not edit the grid, seed, gates, dimensions, or repetitions after output
   is visible.
7. Download `srq_fly_priority2b_memory.zip` and return it for audit.

No dataset, frozen feature cache, or model checkpoint is needed.  A
`PASS_REVIEW_PRIORITY2B` result selects a system backend for later train-only
review; it is not a held-out accuracy result.
