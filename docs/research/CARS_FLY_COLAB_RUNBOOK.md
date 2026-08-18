# CARS-FLY ImageNet-R Colab runbook

Use `notebooks/cars_fly_imagenetr_train_only_colab.ipynb` on a Colab GPU. This
is a train-only feasibility study, not a held-out evaluation.

## What the notebook does

1. clones the locked `feature/cars-fly` branch and verifies the config hash;
2. downloads and verifies the frozen ViT checkpoint;
3. resolves the processed `zaphat206/imagenet-r` Kaggle artifact;
4. runs CARS-FLY mathematical, state, and runner tests;
5. restores or extracts ImageNet-R **training features only**;
6. selects CARS-FLY and raw-Ridge hyperparameters on a deterministic stratified
   training-validation split with seed `2025`;
7. evaluates the locked train-validation controls and exports evidence.

The notebook must never create `test.pt`. Directory indexing may verify the
held-out split's existence, class mapping, and sample count, but no held-out
image is passed through the backbone and no held-out label or feature enters
selection.

## How to run

1. Select **Runtime -> Change runtime type -> T4 GPU**.
2. Open the notebook and edit only the path/source values in cell 2 if needed.
3. Run cells from top to bottom. Do not change seed, search grid, gates,
   checkpoint identity, preprocessing, or model.
4. The extraction cell prints batch progress and one completion line per task.
5. The Phase A cell prints `ANCHOR`, `START`, `TASK`, and `DONE` progress lines.
6. Download `cars_fly_imagenetr_phasea_train_only.zip` from the final cell and
   return it for audit.
7. Stop. A Phase A pass means only that a held-out protocol may be reviewed; it
   does not authorize test evaluation and is not a paper result.

The feature cache under Google Drive is experiment infrastructure. It contains
per-sample training features and must never be described as persistent learner
state or shipped inside a learner checkpoint. CARS-FLY learner state itself is
audited separately and contains only streaming statistics and model metadata.

## Locked identities

- model: `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`;
- pooled feature dimension: `768`;
- checkpoint SHA-256:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`;
- config: `configs/cars_fly_imagenetr_train_only.json`;
- config SHA-256:
  `fd9b117751280aa3369f5db7408448cca4e9e90e82d41c4bfb94beb30e95508c`;
- seed: `2025`;
- tasks: `20` classes-incremental tasks over `200` classes.

