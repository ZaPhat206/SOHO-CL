# Colab CIFAR-100 runbook

Use [tsoho_cifar100_colab.ipynb](../../notebooks/tsoho_cifar100_colab.ipynb), not the older `SOHO_CL.ipynb` benchmark notebook. The old notebook targets Kaggle FLY/SOHO/ImageNet-R and is not a valid T-SOHO selection protocol.

1. In Colab, select **Runtime → Change runtime type → T4 GPU** (or another GPU).
2. Upload the verified `model.safetensors` to Google Drive, for example `MyDrive/T-SOHO/model.safetensors`.
3. Upload/open the Colab notebook. In its first configuration cell, set `REPO_GIT_URL`, `REPO_BRANCH`, and `CHECKPOINT_PATH`. The branch must contain this notebook and `tools/experiment_runner.py`.
4. Run cells in order. It downloads **only CIFAR-100**, verifies the checkpoint, extracts frozen ViT features once, performs train-only selection, then evaluates the locked configuration on test features.
5. Do not use final test metrics to edit `RANKS`, `RIDGE_LAMBDAS`, `VALIDATION_FRACTION`, model, or preprocessing. Rerun selection only when you deliberately define a new protocol.
6. Download `tsoho_cifar100_artifacts.zip`. It contains one directory per final method plus `selection.json`.

The selection cell reserves a deterministic, stratified `VALIDATION_FRACTION` subset from the cached **training** split. It does not open `test.pt`. It selects rank and ridge lambda for `spectral_confusion_code`; the chosen pair is then used unchanged for `raw_ridge`, `random_orthogonal_code`, `truncated_simplex_code`, and T-SOHO to preserve paired fairness.

Expected cost: one full frozen ViT feature extraction pass over CIFAR-100, then small matrix solves on cached 768D features. Cache/output live under `/content`; download artifacts before ending the Colab runtime if you need persistence.
