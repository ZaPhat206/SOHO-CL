# SRQ-FLY Priority 2C Colab runbook

Use `notebooks/srq_fly_priority2c_colab.ipynb` on a Tesla T4.

1. Edit only repository URL/branch and output paths in cell 2.
2. Run cells in order.
3. The correctness gate is CPU/synthetic only.
4. The CUDA benchmark runs 24 isolated workers: one warm-up plus seven measured
   repetitions for Exact FLY, Priority 2B and the implicit-Ridge candidate.
5. Existing workers resume only when config, source, method and profiling
   identities match.
6. Do not edit seed, dimensions, batch size, repetitions or gates.
7. Download `srq_fly_priority2c_memory.zip` and return it for audit.

No dataset, feature cache or checkpoint is needed.  Do not run a held-out test
from this notebook.
