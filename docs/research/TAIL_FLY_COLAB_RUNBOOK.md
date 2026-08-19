# TAIL-FLY ImageNet-R Colab runbook

Use `notebooks/tail_fly_imagenetr_train_only_colab.ipynb` on a Colab T4 GPU.
This is a train-only development study. It never evaluates ImageNet-R test
features or authorizes a held-out run.

## What it does

1. clones the locked `feature/tail-fly` branch and verifies the config hash;
2. verifies the official frozen ViT checkpoint;
3. resolves the processed `zaphat206/imagenet-r` Kaggle artifact;
4. runs all TAIL-FLY synthetic correctness tests;
5. restores the existing train-only ViT feature cache or extracts it once;
6. restores or creates a verified sample-level WTA experiment cache;
7. compares exact FLY, raw Ridge, plain TSVD-FLY, diagonal-only FLY, and
   TAIL-FLY on the same training-validation split;
8. writes resumable unit artifacts to Drive and downloads one evidence ZIP.

The WTA cache can be large. It is disk infrastructure containing sample-level
training codes and is **not** learner state. The notebook reports it separately
and never places it in a TAIL-FLY checkpoint.

## How to run

1. In Colab select **Runtime -> Change runtime type -> T4 GPU**.
2. Open the notebook and edit only source/path values in cell 2 if necessary.
3. Run every cell from top to bottom.
4. Follow the compact progress lines:
   - `WTA CACHE`: create or verify WTA infrastructure;
   - `START` / `DONE`: one resumable exact/raw/rank unit;
   - `TASK`: one class-incremental stage;
   - `RESUME`: a completed unit loaded from Drive.
5. Download `tail_fly_imagenetr_phasea_train_only.zip` from the final cell.
6. Return the ZIP for audit and stop. Do not expose or extract `test.pt`.

If Colab disconnects, rerun from the top. Completed output units resume from
Drive. If interruption occurs while first building the local WTA cache, remove
only that incomplete local WTA directory and rerun; do not overwrite an
incomplete Drive directory.

## Locked identities

- config: `configs/tail_fly_imagenetr_train_only.json`;
- config SHA-256:
  `c49b6a7e813c94d40413dd2d4f8e5e7889fff9c5b1aea0a5e7af046c0913bc04`;
- model: `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`;
- checkpoint SHA-256:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`;
- seed: `2025`;
- split: 20 class-incremental tasks, deterministic 20% train validation;
- representation: dimension 10,000, synaptic degree 300, Top-K ratio 0.3.
