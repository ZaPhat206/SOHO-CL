# Kaggle Phase-4 runbook

1. Commit source, tests, `requirements-kaggle.txt`, configs and notebook; do not commit `data/`, checkpoints, caches or outputs.
2. Upload the repository as a private Kaggle Dataset (recommended for private repos), and upload `model.safetensors` as a separate private Dataset.
3. Add both inputs plus a CIFAR-100 dataset containing `cifar-100/train`, `test`, and `meta` to the notebook. Select a GPU accelerator.
4. Edit only the first notebook configuration cell: `REPO_DIR`, `CIFAR_ROOT`, `CHECKPOINT_PATH`, `CACHE_DIR`, and `OUTPUT_DIR`.
5. Run cells in order: environment, verification, tests/preflight, cache extraction, matrix, aggregate, zip. The notebook writes the cache under `/kaggle/working` and validates metadata before every run.
6. On interruption rerun the matrix cell with `--resume`; each run has a `progress.pt` task-boundary checkpoint until it finishes. Retain `/kaggle/working/outputs` as a Dataset version or download the ZIP.
7. A successful run has `config.json`, `environment.json`, `metrics.json`, `task_accuracies.csv`, `accuracy_matrix.csv`, `state_bytes.csv`, `timing.csv`, `code_diagnostics.json`, and `run.log` per run. Return the ZIP and these files for analysis.

Kaggle inputs are read-only: copy repo source to `/kaggle/working` before writing; caches and outputs stay under `/kaggle/working`. If GitHub Internet is enabled, cloning is an alternative; do not hard-code a token. Without access, use the private Dataset upload path above.
