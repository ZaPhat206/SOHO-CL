# REPO AUDIT — SOHO-CL và kế hoạch T-SOHO

**Phạm vi.** Audit tĩnh repository tại `SOHO-CL` (và baseline gốc tại `../LAB_FLY`) ngày 2026-08-12. Không thay đổi implementation và không chạy training/evaluation. Ngoại lệ duy nhất trong workspace là chính tài liệu audit này. `notebooks/kaggle_runner.py` đã có thay đổi trước audit (`git status`); tài liệu này không sửa file đó.

## Kết luận ngắn

SOHO hiện tại không exemplar-free theo nghĩa đặt ra cho T-SOHO: `SOHOCL` không giữ ảnh, nhưng giữ mọi backbone embedding lịch sử và nhãn trong `memory_features`/`memory_labels`, rồi chiếu lại toàn bộ khi OLDA đổi `R`. Ngược lại, FlyCL đã có streaming sufficient statistics `G_global`, `Q_global` cho feature sau FlyHash cố định.

T-SOHO nên được đặt ở **low-rank structured projection của không gian label/logit**, trực tiếp trên frozen feature `x`, không phải là một phép xoay OLDA/ETF hoặc transport qua Top-K. Khi `G` và `Q` tích luỹ trên raw frozen features, đổi code `E` chỉ cần giải lại deterministic từ sufficient statistics; không cần historical feature.

## 1. Data flow chính xác: dataset → logits

```text
CLI args
  → utils/data_utils.load_dataset
  → torchvision dataset/ImageFolder + deterministic class permutation
  → {task_id: DataLoader[Subset]}
  → utils/train_utils.feature_extract
  → frozen timm ViT (num_classes=0) → embedding x ∈ R^768

FlyCL: x → fixed sparse FlyHash projection → per-sample positive Top-K/WTA
       → X_fly → cumulative G=XᵀX, Q=XᵀY → W_o=(G+λI)^−1Q
       → test x → same FlyHash + same WTA → logits=X_fly W_o → global argmax

SOHOCL: x → IncrementalOLDA statistics → task-current R
        → L2 normalize x → xRᵀ → fixed dense sparse-Rademacher W
        → per-sample positive Top-K/WTA → X_soho
        → G,Q rebuilt from *all retained raw backbone embeddings* under current R
        → W_o=(G+λI)^−1Q
        → test x → current R/W/WTA → logits=X_soho W_o → global argmax

T-SOHO proposed: x → cumulative G_x=Σxᵀx, Q_x=ΣxᵀY (+ class sums/counts)
        → deterministic E_t from class-confusion geometry
        → P_t=(G_x+λI)^−1Q_xE_tᵀ
        → z=xP_t; logits_c=2zᵀe_c−||e_c||² → global argmax
```

`task_id` is used only to select a train/test loader and to index the reported accuracy matrix. Neither FlyCL nor SOHOCL passes it into their predictor; evaluation is class-incremental over all `num_classes` output columns.

## 2. Mã nguồn và vai trò

| Thành phần | Đường dẫn | Vai trò thực tế |
|---|---|---|
| Entry point / task loop | `main.py` | parses CLI, loads data/backbone/agent; loops `train_task(t)` then evaluates every `sub_task ≤ t`. |
| Dataset & preprocessing | `utils/data_utils.py` | CIFAR-100/CUB/VTAB/ImageNet-R/etc.; resize/center crop 224, ToTensor, optional `vit` or `resnet` normalization; random class partition. |
| Frozen extractor | `models/backbone.py`, `utils/train_utils.py` | `timm.create_model(..., pretrained=True, num_classes=0)`, `.eval()`, then full-loader feature extraction under `no_grad`. |
| Continual abstraction | `methods/base_cl.py` | minimal `BaseCL` interface. |
| SOHO task/update | `methods/sohocl.py` | holds historical embeddings, updates OLDA, reprojects all historical samples, chooses GCV λ, rebuilds and solves Ridge. |
| OLDA | `models/soho.py:IncrementalOLDA` | normalized per-class sums/counts, global sum/count and within-class scatter `S_w`; generalized LDA; discriminative plus null-space basis. |
| ETF/Procrustes | `models/soho.py:compute_projection` | builds an `(N−1)×N` simplex ETF, then rotates OLDA discriminative basis with orthogonal Procrustes; enabled unless `--no_etf`. |
| SOHO sparse projection / Top-K | `models/soho.py:SOHO.forward` | `R` from OLDA, then a fixed **dense stored** Rademacher matrix `W` (zeros/±1), followed by sample-dependent positive WTA. |
| Fly baseline / Top-K | `models/flyhash.py`, `methods/flycl.py` | fixed FlyHash matrix converted to sparse CSC, positive WTA, streaming G/Q and Ridge. |
| Ridge/GCV | `methods/flycl.py`, `methods/sohocl.py` | duplicate GCV helper (SVD of current feature set); Cholesky solve of `(G+λI)W=Q`. |
| Evaluation/metrics | `main.py`, `utils/metrics.py` | per-seen-task accuracy matrix, AA/A_T/LA/forgetting/BWT, timing, peak CUDA allocation. |
| Experiment profiles | `notebooks/kaggle_runner.py` | runnable Kaggle profile/command constructor for FlyCL and SOHOCL; currently user-modified. |
| Nominal configuration | `README.md`, `configs/flycl_cifar100.yaml` | README advertises `--config`; file is empty and `main.py` has no YAML/config loader. |

## 3. Learner state qua task

| Method | Tồn tại qua task | Exemplar-free? | Ghi chú |
|---|---|---|---|
| FlyCL | fixed projection; `G_global` `(10000,10000)`; `Q_global` `(10000,C)`; `Wo`; scalar/hyperparameters | Yes wrt images/features | No checkpoint serialization is implemented. GCV itself sees only the current task. |
| SOHOCL | all historical `memory_features` `(N_seen,768)` and `memory_labels`; SOHO `R`, `W`, `Wo`; OLDA `class_sums`, `class_counts`, `global_sum/count`, `S_w` | **No** wrt historical features | Does not retain raw images. `G_global/Q_global` are local tensors rebuilt each task, not stored as agent state. |
| T-SOHO target | `G_x` `(D,D)`, `Q_x` `(D,C)`, class sums/counts (optionally normalized sums), observed-class mask, λ/config, current deterministic `E/P` or recomputable equivalents | Yes | No raw images and no per-example historical feature/label tensors. |

For FP32 at `D=768`, T-SOHO statistics are approximately 2.25 MiB for `G_x` plus 0.29 MiB for `Q_x` at `C=100`, excluding model; this is qualitatively different from SOHOCL's growing replay. This estimate excludes optional cached `P/E` and uses binary MiB.

## 4. SOHO implementation vs reported/proposed formulation

No SOHO technical report/manuscript is present in this repository. The comparison below is therefore against the supplied formulation and claims visible in code comments, not a source-of-truth SOHO report.

| Topic | Observed SOHO | T-SOHO formulation / implication |
|---|---|---|
| Representation statistics | OLDA statistics are incremental, but classifier statistics are not transportable after `R` changes. | Accumulate `G_x=Σxᵀx`, `Q_x=ΣxᵀY` before any dynamic transform. |
| Projection | Dynamic orthogonal/orthonormal `R`, followed by fixed random expansion and dynamic WTA. | `P=(G+λI)^−1QEᵀ` is a learned, label-geometry-dependent low-rank map. |
| Class structure | ETF/Procrustes acts on OLDA feature basis; it supplies no label-code classifier. | Compute graph Laplacian `L` from sufficient class means; select rank `r<C_seen−1` and code `E∈R^(r×C)`, `EEᵀ=I`. |
| Classifier | Multiclass one-hot Ridge `W_o`. | Global analytic code Ridge plus prototype decoder; no Task-ID. |
| Memory | Replay of every frozen feature is required because `R` and WTA change. | Exact streaming sufficient stats remain valid because feature space is fixed. |

### Required algebraic checks before claiming novelty

Let `W_R=(G+λI)^−1Q`, with the same `G,Q,λ` as the raw-feature Ridge baseline. Then

`P=(G+λI)^−1QEᵀ = W_R Eᵀ`, so the score-dependent part of T-SOHO decoding is

`2xPE = 2xW_R(EᵀE)`.

Thus `W_TSOHO = W_R EᵀE` is correct only as the linear score component (with a factor 2 under the specified decoder); the full logits also include `−||e_c||²`. Tests must check it numerically.

If `E` is full orthogonal, `EᵀE=I`, so isotropic Ridge is merely a change of basis (modulo decoder bias). For a full simplex/centered ETF, `EᵀE=I−11ᵀ/C`; it subtracts the same per-sample mean score from every class. With equal ETF column norms, argmax is equivalent to raw Ridge. Therefore the experimental hypothesis must use strict low rank `r<C_seen−1` and demonstrate an effect beyond this equivalence.

Dynamic Top-K is sample-dependent and nonlinear. In general there is no single matrix `T` for which `TopK(xR_tᵀWᵀ)=TopK(xR_{t−1}ᵀWᵀ)T` for every `x`; SOHO's replay/reprojection is therefore consistent with its own operator, but cannot be replaced exactly by linear transport of its existing `G/Q`.

## 5. Minimal insertion point for `t_soho`

Keep old SOHO intact. Add a sibling learner rather than modifying `SOHO`, `SOHOCL`, or FlyCL:

1. `methods/tsohocl.py`: owns frozen-feature `G/Q`, class sufficient statistics, graph/code construction, deterministic Ridge solve and prototype logits.
2. `models/tsoho.py` (or a small `utils/tsoho_math.py`): pure, unit-testable functions for means → graph Laplacian → canonical spectral code → decode. It must have no OLDA, random expansion or WTA dependency.
3. `main.py`: add explicit `t_soho` choice, T-SOHO CLI parameters (`transport_rank`, `graph_tau`, `ridge_lambda`/GCV policy, graph weighting/normalization), and instantiate the sibling learner.
4. `notebooks/kaggle_runner.py`: only after core CLI works, add a distinct profile and fair comparison table; do not reuse SOHO hyperparameters implicitly.
5. Tests and documented commands: add before enabling broad experiments.

This leaves `sohocl` behaviour and its output path unchanged.

## 6. Reproduction evidence: FLY-CL, preprocessing, checkpoints

The most authoritative local FLY-CL commands are:

```bash
cd LAB_FLY/scripts
./test_cifar.sh  # CIFAR-100, 10 tasks, ViT-B/16, D=768, m=10000, degree=300, k=0.3, seed=1993, vit normalize, λ=10^6..10^9
./test_cub.sh    # CUB-200-2011, 10 tasks, same core settings, seed=2023
./test_vtab.sh   # VTAB, 5 tasks / 50 classes, same core settings, seed=2023
```

Equivalent commands are the single `python main.py ...` lines in those scripts. `LAB_FLY/readme.md` records the intended environment: Python 3.9, PyTorch 1.13.1/torchvision 0.14.1/CUDA 11.7, NumPy <2, timm 0.9.16, tqdm and scipy. `LAB_FLY/pretrained_model/download.sh` downloads the ViT-B/16 IN-21K `.npz`; the original `LAB_FLY/models/load_model.py` uses `timm` pretrained ViT for `vit_base_patch16_224` and has an alternative local ResNet checkpoint path.

SOHO-CL changed loader details: CIFAR uses `./data` plus a possible symlink, and `models/backbone.py` accepts arbitrary timm names. `notebooks/kaggle_runner.py` is the only SOHO-CL experiment launcher; it profiles CIFAR/CUB/ImageNet-R and uses Kaggle root `/kaggle/input/datasets/zaphat206`. No `torch.save`, `torch.load`, checkpoint resume, requirements file, test runner, or valid YAML-driven experiment command exists in SOHO-CL.

## 7. Implementation plan — small commits, after approval

1. **`test: establish deterministic math fixtures`** — add CPU synthetic tests for one-hot sufficient-stat updates, raw Ridge closed form, no Task-ID global scoring, and seeded task split. No T-SOHO path yet.
2. **`feat: add pure T-SOHO label-geometry primitives`** — graph from counts/sums, Laplacian, rank validation, canonical spectral-code routine and prototype decoder. Define handling of absent classes and degenerate eigenvalues explicitly.
3. **`test: prove/guard transport identities`** — numeric tests for `P=W_REᵀ`, decoded score equality to `2xW_REᵀE−norm²`, full orthogonal invariance, full simplex argmax equivalence, and strict-low-rank non-equivalence fixture.
4. **`feat: implement streaming TSOHOCL`** — feature extraction once per current task; update only `G,Q,class_sums,class_counts`; recompute `E,P` deterministically; never append an example tensor. Add `state_summary()` for auditable memory accounting.
5. **`feat: expose t_soho CLI and reproducible configs`** — wire `main.py`, make configuration loading real or remove the misleading README option, persist task-class order/args in a result artifact, and add a T-SOHO profile without touching existing SOHO defaults.
6. **`test: add integration smoke matrix`** — CPU toy dataset and a tiny frozen mock backbone; run FlyCL, SOHOCL and T-SOHO for ≥2 tasks, assert old SOHO outputs remain unchanged under a fixed seed and T-SOHO state contains no per-example replay.
7. **`docs: protocol and result schema`** — record shared splits/preprocessing/seeds/lambda policy and generate a comparison table for FLY-CL, SOHO, raw-feature Ridge, T-SOHO plus rank/τ ablations. Only then launch research-scale jobs.

## 8. Test plan

| Level | Test | Pass criterion |
|---|---|---|
| Unit | streaming update | Sequential-batch `G,Q,sums,counts` equal a concatenated reference to tolerance. |
| Unit | graph/code | `L` symmetric, `E Eᵀ≈I`, selected rank valid for observed classes; deterministic ordering/sign convention. |
| Unit | algebra | Exact identities above hold; full orthogonal / full ETF fixtures do not create a claimed gain. |
| Unit | decoder | `argmax` uses all observed global classes and accepts no task identifier. |
| Unit | state audit | T-SOHO has no list/tensor with leading dimension proportional to number of seen examples; state is bounded by `O(D²+DC)`. |
| Regression | existing FlyCL/SOHO | Existing smoke fixture outputs/metric layout preserved; `sohocl` imports untouched. |
| Integration | two/three task toy stream | T-SOHO trains/evaluates with no replay; unseen/observed-class policy behaves as documented. |
| Reproduction | fair small run | Same frozen model, task split, transforms, `D`, batch size, seeds and λ search policy across raw Ridge/T-SOHO; report all runtime/memory inclusions. |
| Research run (post-approval) | rank/τ/normalization ablation | Include `r=C_seen−1` and full-rank controls specifically to falsify transform-only novelty. |

Do not use SOHO's `expand_dim`, density, or Top-K as hidden advantages for raw Ridge/T-SOHO unless the protocol explicitly studies those representation changes. Peak CUDA allocation is not a sufficient learner-state metric; report retained-state bytes separately from temporary solver/workspace and frozen backbone.

## 9. Missing information / inconsistencies requiring resolution

1. There is no SOHO report in the repo, no dependency lock/`requirements.txt`, no test suite, no checkpoint/resume support, and no stored class-order/result artifact. Exact SOHO claimed results cannot be independently reproduced from this checkout.
2. `README.md` says `python main.py --config configs/flycl_cifar100.yaml`, but parser has no `--config` and that YAML is empty. This command fails as documented.
3. `main.py` defaults to 20 tasks; Kaggle profiles and FLY paper scripts use 10 (or VTAB 5). SOHO/Fly results are not comparable unless task partition and all preprocessing are recorded.
4. SOHO's task-current `R` changes after every update and its WTA is nonlinear; its classifier is rebuilt from replay, so it is not a streaming-statistics baseline despite incremental OLDA.
5. SOHO `olda_dim` is passed into `SOHO`, but `output_dim` is `expand_dim`; `compute_projection` caps `R` by input dimension. The intended distinction between `olda_dim`, `expand_dim`, and the report's claimed sparse projection dimensionality needs formal documentation.
6. Current GCV policy is not matched: FlyCL uses current-task projected data, while SOHO samples up to 3000 retained examples (random sample without a generator). A fair T-SOHO/raw Ridge protocol must choose one shared λ-selection policy or explicitly call it an oracle/validation condition.
7. No policy is defined for spectral graph construction when a new class has few samples, for disconnected graphs/eigenvalue multiplicity, code sign/basis ambiguity, or columns corresponding to unseen classes. These are necessary for deterministic streaming behavior.
8. The requested formula uses class means or `Q/counts`; current `Q` is one-hot cross-covariance. With raw `x`, `Q[:,c]/n_c` equals an unnormalized feature mean; if graph distance is intended on L2-normalized backbone means, separate normalized class sums must be retained.
9. The decoder's `−||e_c||²` makes code-column norms a modelling decision. Spectral eigenvectors alone need not have equal column norms, so its bias can confound a purported geometry benefit. Specify whether codes are normalized, whether a bias is fitted, and whether it is included in raw-Ridge control.
10. `data_utils.py` names `data_augmentation`, but applies no stochastic augmentation—only normalization. CIFAR has profile value `none`, whereas original FLY reproduction scripts use `vit`; this needs an explicit fair-preprocessing decision.
11. Repository worktree already has a modified Kaggle runner, so experiments should not silently treat it as a clean baseline. Git currently reports it modified.

## Anticipated files to change after approval

- `methods/tsohocl.py` — new streaming learner.
- `models/tsoho.py` **or** `utils/tsoho_math.py` — new deterministic label-geometry primitives (choose one convention before implementation).
- `main.py` — method registration and explicit CLI/config plumbing.
- `notebooks/kaggle_runner.py` — optional, after core path; preserve its user changes.
- `configs/t_soho_*.yaml` plus either a real config loader or documented CLI scripts.
- `tests/test_tsoho_math.py`, `tests/test_tsohocl_streaming.py`, `tests/test_regression_baselines.py` — new tests.
- `README.md` — valid commands, reproducibility protocol and state/memory semantics.

Files that should not need modification for a minimal T-SOHO path: `models/soho.py`, `methods/sohocl.py`, `models/flyhash.py`, and `methods/flycl.py`.
