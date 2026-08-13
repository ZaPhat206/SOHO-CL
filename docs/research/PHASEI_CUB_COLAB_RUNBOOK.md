# Phase I CUB Colab runbook

Open `notebooks/phasei_cub_train_only_colab.ipynb` and run cells in order on a
T4 GPU. Edit only repository/Drive paths in cell 2.

The notebook downloads `zaphat206/cub-200-2011`, verifies every image against
the locked processed-dataset identity, obtains the official frozen ViT
checkpoint, and extracts only the 5,994 training embeddings. It deliberately
does not create `test.pt` or evaluate test accuracy.

The selection cell prints one short line per prepared task and one line per
candidate. It can safely resume: the gate cache and every completed candidate
are stored under Google Drive. Do not delete or edit these caches while the
cell is running. GPU utilization can be low during analytic solves; this is
normal because the matrices are small and candidate orchestration is partly
CPU-bound.

At completion, download and return
`phasei_cub_train_only_selection.zip`. If the final status is FAIL, stop; do
not adjust the grid and do not run the held-out test. If it is PASS, still stop
for artifact audit and explicit authorization.
