# SRQ-FLY D3 CUB Colab runbook

Use `notebooks/srq_fly_cub_d3_train_only_colab.ipynb` on a Colab T4 GPU. This
is a nested **train-only** replication, not a CUB test run.

1. Push branch `feature/srq-fly-d3-cub-replication` before opening Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. In cell 2, edit path/source values only. Do not change seed, dimensions,
   lambda grids, split fractions, or gates.
4. Run all cells in order. The notebook first verifies the repository config,
   checkpoint, processed dataset identity, tests, and train-only feature cache.
5. Cache cells print bounded copy/extraction progress. The experiment cell
   prints `CACHE`, `INNER START/DONE`, `LOCKED`, `OUTER`, and `TASK` lines.
6. Interruptions are resumable when the exact same code/config/cache identities
   are retained. A stale unit error means use a new output directory rather
   than mixing protocols.
7. Download `srq_fly_cub_d3_train_only.zip` and return it for audit.
8. Stop even if the status is `PASS_REVIEW_D3`. Do not create or evaluate
   CUB `test.pt`.

The feature cache and both WTA caches contain sample-level training data and
are experiment infrastructure. They are stored separately on Drive, excluded
from the ZIP, and must never be called learner state or bundled in a learner
checkpoint.
