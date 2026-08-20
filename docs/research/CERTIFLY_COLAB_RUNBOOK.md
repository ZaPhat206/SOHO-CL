# CertiFLY ImageNet-R Colab runbook

Use `notebooks/certifly_imagenetr_train_only_colab.ipynb` on a Colab T4 GPU.
This is the locked Q1 train-only feasibility study. It does not evaluate or
authorize access to held-out ImageNet-R features.

## What it does

1. clones `feature/certifly` and verifies the locked config identity;
2. verifies the exact frozen ViT checkpoint;
3. resolves the processed `zaphat206/imagenet-r` artifact;
4. runs all CertiFLY synthetic correctness and protocol tests;
5. restores or extracts training embeddings only;
6. restores or creates the exact FLY WTA-code experiment cache;
7. compares matched exact FLY, raw Ridge, fixed-int8 CertiFLY, and two
   adaptive int8/int16 certificate budgets on one paired train-validation
   stream;
8. exports resumable evidence to Drive and downloads one ZIP.

The feature and WTA caches contain sample-level training data. They are disk
experiment infrastructure, not learner state, and must never be packaged in a
CertiFLY checkpoint or used to claim an exemplar-free checkpoint.

## How to run

1. Select **Runtime -> Change runtime type -> T4 GPU**.
2. Open the notebook. Edit only repository/checkpoint/Drive paths in cell 2.
3. Run every cell from top to bottom.
4. Follow the compact progress lines:
   - `WTA CACHE`: restore, verify, or build code-cache infrastructure;
   - `START` / `DONE`: one resumable control or candidate;
   - `TASK`: one class-incremental stage, certificate ratio, block widths,
     state bytes, and elapsed time;
   - `RESUME`: a completed unit loaded from Drive.
5. Download `certifly_imagenetr_q1_train_only.zip`, return it for audit, and
   stop. Do not expose or evaluate `test.pt`.

If Colab disconnects, rerun from the first cell. Completed method units resume
from Drive. If WTA creation was interrupted before `metadata.json` was written,
delete only that incomplete local WTA directory and restore the known-good
Drive cache again.

## Locked protocol

- config: `configs/certifly_imagenetr_train_only.json`;
- config SHA-256:
  `57ccd8969b6d0b8783f302940bce445d85bb11ccc48d2aee28952927c1bfa0ee`;
- model: `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`;
- checkpoint SHA-256:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`;
- seed: `2025`;
- stream: 200 classes, 20 tasks, deterministic 20% training validation;
- FLY representation: dimension 10,000, synaptic degree 300, Top-K 0.3;
- Q1 success gate: at most 0.5 percentage point below exact FLY and at most
  25% of exact-FLY persistent learner-state bytes, with the numerical
  certificate and solver residual valid.

A Q1 pass is a development result, not evidence of strict reproduction or a
paper-level claim.
