# Phase 3 — baseline integration and FLY-CL CIFAR-100 reproduction

Status: **baseline integration PASS on synthetic stream; CIFAR-100 reproduction BLOCKED before execution.** No claim of a reproduced FLY-CL result is made in this document.

## Scope and exclusions

Implemented only `streaming_raw_ridge`, a global exemplar-free raw-backbone-feature Ridge baseline. No random code, ETF code, spectral code, confusion graph, or T-SOHO model was added. No CUB or ImageNet-R command was run.

## Locked comparison configuration — before run

The reported FLY-CL reference is Table 1 of the local paper text (`../fly_paper.txt`): ViT-B/16 on CIFAR-100, overall/average incremental accuracy `93.89±0.12` and average training time `19.07±0.07 s` (the table extraction places the Fly-CL row under CIFAR-100). The reproduction source of truth for operational arguments is `../LAB_FLY/scripts/test_cifar.sh`.

| Item | Reported/source FLY-CL | Current `SOHO-CL --method flycl` | Status |
|---|---|---|---|
| Dataset/classes/tasks | CIFAR-100 / 100 / 10 | same when invoked below | matched |
| Backbone | `vit_base_patch16_224`, frozen, output 768 | same `timm.create_model(..., pretrained=True, num_classes=0)` | source-level matched; exact cached checkpoint not yet verified |
| Preprocessing | script requests `--data_augmentation vit` | same; resize/crop to 224 for CIFAR then `Normalize([.5]*3,[.5]*3)` | source-level matched |
| Class order/task split | Python `random.sample(range(100),100)` after seed 1993 | same helper and seed path | source-level matched; manifest cannot be emitted before run |
| Projection distribution | 10,000 rows; choose 300 of 768 without replacement; iid normal nonzeros; CSC | `FlyHash` has the same construction then CSC conversion | source-level matched |
| Top-K | positive Top-3000 (`coding_level=.3`) per sample | same | matched |
| Ridge selection | current-task GCV over `10^6,…,10^9` | same formula/range on current projected task | matched |
| Solver | Cholesky solve of cumulative `G,Q` | Cholesky solve of cumulative `G,Q` | matched |
| Evaluation | after task `t`, evaluate each `sub_task≤t`; global 100-column logits | same outer loop and 100-column FlyCL head | matched |
| Runtime environment | README specifies Py3.9, torch 1.13.1, torchvision .14.1, CUDA 11.7, timm .9.16 | `timm .9.16` installed after the initial failed command; actual runtime is Python 3.13.5, torch 2.12.0+cpu, torchvision .27.0+cpu, no CUDA device | material mismatch; CPU reproduction not yet run |
| Dataset/checkpoint availability | script assumes assets available/downloadable | CIFAR-100 is present at `../processed_datasets/cifar-100/{meta,train,test}`; use `--root ../processed_datasets`. Exact ViT checkpoint/cache provenance remains unverified. | dataset resolved; checkpoint provenance pending |

`configs/streaming_raw_ridge_cifar100.json` locks the raw-Ridge counterpart's frozen backbone, preprocessing, 10-task class-incremental stream, seed 1993, batch size 128, and the same **current-task train-only GCV range**. It intentionally has no FlyHash projection/Top-K parameters because raw Ridge is the unprojected control.

## Baseline contract

For feature batch `X:(B,D)` and labels mapped into sorted observed global class IDs:

```text
G ← G + XᵀX                         # (D,D)
Q ← expand_columns(Q) + XᵀY_seen    # (D,C_seen)
W ← solve(G + λI, Q)                # no explicit inverse
logits(X) = XW                      # all seen-class columns; no Task-ID input
```

Persistent tensors are `G_global`, `Q_global`, and derived inference weights `Wo`; permitted non-tensor metadata is `class_ids`, its class-to-column mapping, dimensions, GCV range, and the chosen scalar λ. `Q_global` grows only by class columns. The learner stores no image, dataset index, path, historical embedding, historical per-sample label, or replay list. `Wo` is retained solely as the derived global classifier; it can be recomputed from `G,Q,λ`.

`persistent_state_summary()` reports exact names, shapes, and allocated tensor bytes after each task; `assert_exemplar_free_state()` rejects unexpected persistent tensors and sample-shaped leading dimensions. The test below verifies the final streaming classifier against a batch oracle on a two-update toy stream.

## Commands executed

Integration/unit validation:

```bash
cd D:/lab/FLY/SOHO-CL
python -m pytest -q
```

Final result after all Phase 3 changes: `8 passed in 3.42s`.

## Step 1 — compatibility preflight

Dataset preflight used the frozen protocol seed `1993`, root argument `../processed_datasets`, and the production-equivalent torchvision dataset root `./data`. The current loader tries to make `./data/cifar-100-python → ../processed_datasets/cifar-100` as a symbolic link. The Windows sandbox lacks symlink privilege (`WinError 1314`), so a local NTFS **junction** was created at `D:\lab\FLY\SOHO-CL\data\cifar-100-python` pointing to `D:\lab\FLY\processed_datasets\cifar-100`. It neither copies nor changes the dataset. That is the concrete path resolved by torchvision.

The first inline preflight also encountered a Windows multiprocessing limitation (`num_workers=8` attempts to import `<stdin>`). Dataset inspection therefore used the same dataset and transform with `num_workers=0`; this is a preflight-only execution detail, not a training-protocol change. `main.py` is a real file entry point and does not have the `<stdin>` import path.

| Check | Result |
|---|---|
| CIFAR train/test load | PASS — 50,000 / 10,000 samples; 100 classes. |
| Class mapping | PASS — canonical torchvision CIFAR-100 mapping, IDs 0–99: `apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle, bowl, boy, bridge, bus, butterfly, camel, can, castle, caterpillar, cattle, chair, chimpanzee, clock, cloud, cockroach, couch, crab, crocodile, cup, dinosaur, dolphin, elephant, flatfish, forest, fox, girl, hamster, house, kangaroo, keyboard, lamp, lawn_mower, leopard, lion, lizard, lobster, man, maple_tree, motorcycle, mountain, mouse, mushroom, oak_tree, orange, orchid, otter, palm_tree, pear, pickup_truck, pine_tree, plain, plate, poppy, porcupine, possum, rabbit, raccoon, ray, road, rocket, rose, sea, seal, shark, shrew, skunk, skyscraper, snail, snake, spider, squirrel, streetcar, sunflower, sweet_pepper, table, tank, telephone, television, tiger, tractor, train, trout, tulip, turtle, wardrobe, whale, willow_tree, wolf, woman, worm`. |
| One train batch | PASS — images `(128, 3, 224, 224)`, `torch.float32`; labels `(128,)`, `torch.int64`. |
| Preprocessing | `Resize(224, bicubic) → CenterCrop(224) → ToTensor() → Normalize(mean=(.5,.5,.5), std=(.5,.5,.5))`. |
| Task count / task-0 classes | 10 tasks; task 0 IDs `[61,79,33,57,4,14,21,42,44,19]` (5000 train / 1000 test samples). |
| Complete class order | `[61,79,33,57,4,14,21,42,44,19,51,73,45,89,35,85,39,56,0,24,65,29,9,18,13,95,41,80,96,32,15,49,22,99,63,68,1,62,46,59,23,60,7,86,3,27,67,69,50,92,31,98,76,84,97,93,43,16,30,83,12,5,66,72,48,78,54,81,53,26,20,94,74,47,88,38,90,10,36,11,40,52,64,87,91,6,8,55,77,82,25,75,17,28,70,2,58,71,37,34]`. |
| Projection / WTA / Ridge | `m=10000`; each projection row chooses 300 of 768 coordinates without replacement and fills them with iid `torch.randn` values (other values zero), converted to CSC; positive Top-3000 (`0.3`); current-task GCV candidates `10^6,10^7,10^8,10^9`, then Cholesky solve. |
| Batch/device/environment | batch 128; CPU; Python 3.13.5; torch 2.12.0+cpu; torchvision 0.27.0+cpu; timm 0.9.16. |

Checkpoint preflight **FAIL**. The source-level backbone name is `vit_base_patch16_224`; under the installed timm it resolves to Hugging Face model ID `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`. There is no local cached weight file and network access to Hugging Face is blocked. Relevant exact traceback tail:

```text
httpcore.ConnectError: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions
...
huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.
```

Consequently checkpoint load did not succeed, no `load_state_dict` occurred, missing/unexpected keys are not available, and feature dimension/feature finiteness cannot be verified with pretrained weights. This is a checkpoint-availability failure, not a CPU correctness failure.

Exact locked FLY-CL reproduction command attempted:

```bash
cd D:/lab/FLY/SOHO-CL
python main.py --method flycl --dataset CIFAR-100 --root ../processed_datasets --num_classes 100 --num_tasks 10 --model_name vit_base_patch16_224 --embedding_dim 768 --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 --seed 1993 --batch_size 128 --gpu 0 --data_augmentation vit --ridge_lower 6 --ridge_upper 10
```

The first attempt (before `timm` was installed) stopped before dataset loading/training with exit code 1:

```text
Traceback (most recent call last):
  File "D:\\lab\\FLY\\SOHO-CL\\main.py", line 10, in <module>
    from models.backbone import load_model
  File "D:\\lab\\FLY\\SOHO-CL\\models\\backbone.py", line 1, in <module>
    import timm
ModuleNotFoundError: No module named 'timm'
```

Locked raw-Ridge command (not run because the shared environment blocker applies):

```bash
cd D:/lab/FLY/SOHO-CL
python main.py --config configs/streaming_raw_ridge_cifar100.json --root ../processed_datasets --gpu 0
```

## Reproduction result and stopping rule

Step 2 CPU smoke test and Step 3 full compatibility reproduction were **not started**, as required by the checkpoint preflight gate. Since no CIFAR-100 task was trained/evaluated, the following metrics are **not available** for either baseline: per-task accuracy, final accuracy, average incremental accuracy, forgetting, update time, inference time, peak runtime memory, and persistent learner-state bytes on CIFAR-100.

The required ±0.5 percentage-point comparison against the reported `93.89%` cannot be evaluated. This phase must remain blocked—not called reproduced—until the specified environment, exact checkpoint provenance, CIFAR-100 data, and a successful run manifest are available. No hyperparameter was changed and no test-set tuning was attempted.

## Memory terminology

| Category | Meaning | Phase 3 handling |
|---|---|---|
| Experiment cache on disk | Downloaded CIFAR files, pretrained-weight cache, logs/results, and any future feature cache. | Not learner state. A feature cache must not be required by `streaming_raw_ridge` training/resume if it is called exemplar-free. No such cache was created. |
| Runtime memory | CUDA/CPU allocations while extracting features, projecting, selecting λ, solving, or evaluating. | `compute_memory_footprint` reports peak CUDA allocated memory only; main also reports end-to-end and post-feature-extraction inference timing. No CIFAR measurement exists yet. |
| Persistent learner state | Objects intentionally retained after task completion for future learning/inference. | Raw Ridge inventory is `G_global`, `Q_global`, `Wo` plus bounded class mapping/metadata; images and per-example features/labels are forbidden. FLY-CL inventory additionally includes its fixed projection matrix. |

## Next permitted action

Use the verified local CIFAR-100 path in the two locked commands. Before treating results as a reference reproduction, obtain/approve the paper-compatible GPU environment and verify exact checkpoint provenance; the currently installed runtime is CPU-only and version-mismatched. If a comparison run is nevertheless approved and Fly-CL differs by more than 0.5 percentage point from 93.89%, stop, record the observed metric and diagnose checkpoint/version, preprocessing, class-order, split, projection RNG, solver/dtype, and metric-definition differences before any further experiment.
