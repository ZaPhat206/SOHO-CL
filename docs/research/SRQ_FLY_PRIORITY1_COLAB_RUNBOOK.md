# SRQ-FLY Priority-1 Colab runbook

Use `notebooks/srq_fly_priority1_colab.ipynb` on a T4 GPU.

The notebook first runs correctness tests, then the six-process synthetic
system benchmark at width 10,000. Only if that benchmark passes does it extract
or reuse CIFAR-100 **training features only** and launch the six isolated
train-validation ablation workers. Each worker prints one `TASK` line per
stage. Completed JSON units remain under the configured output directory.

Do not expose or create `test.pt`, edit the locked protocol, relax a failed
gate, or report train-validation accuracy as a paper test result. Return
`srq_fly_priority1_train_only.zip` for audit. Packed int4 and error feedback are
deliberately outside this notebook.

The selected optimized backend is blocked rank-update QR with a locked panel
size of 128. The system cell always prints the complete gate table, method
timings, persistent bytes, and CUDA peaks before stopping on failure, so a
failed engineering gate is diagnosable without another run.
