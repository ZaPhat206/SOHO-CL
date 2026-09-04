# SRQ-FLY Priority 4 Colab runbook

Run `notebooks/srq_fly_priority4_task_frequency_colab.ipynb` from top to bottom
on a T4 GPU.

The notebook extracts or reuses **CIFAR training features only**, builds five
WTA caches, and runs 10 paired resumable units: five seeds times 10- and
20-task schedules. It never creates or reads `test.pt`.

## Expected flow

1. Edit only path/source values in cell 2.
2. Run the source-hash and correctness gates.
3. Download the frozen checkpoint and processed CIFAR-100 source.
4. Restore or extract `train.pt`; confirm `test.pt absent`.
5. Run the locked Priority-4 cell. Each unit prints `START`/`DONE`; rerunning
   the cell restores source-matched completed units.
6. Inspect the paired table and plots.
7. Download `srq_fly_priority4_task_frequency_train_only.zip` and return it for
   audit.

Do not change the five seeds, task schedules, split, representation, Ridge,
gates, or method after seeing output. A `STOP_PRIORITY4_TASK_FREQUENCY` result
must still be returned unchanged.

The first unit for each projection seed creates a roughly 900 MB WTA cache;
the second schedule reuses it. These five caches are experiment infrastructure
and are deliberately excluded from the ZIP and learner-state accounting.
