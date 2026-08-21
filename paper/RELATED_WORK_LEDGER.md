# SRQ-FLY related-work ledger

This ledger records the primary source used for each literature statement and
the boundary between that work and SRQ-FLY. It is an audit aid, not a claim
that the literature review is exhaustive.

| Key | Primary source | Relevant contribution | Boundary for SRQ-FLY |
|---|---|---|---|
| `zhuang2022acil` | [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract-Conference.html) | Recursive analytic CIL without historical data; joint/incremental equivalence under its assumptions. | SRQ-FLY compresses the expanded Ridge system and does not claim ACIL's entire architecture or privacy theorem. |
| `mcdonnell2023ranpac` | [NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html) | Frozen random nonlinear expansion and prototype decorrelation for pretrained-model CL. | SRQ-FLY uses the repository's FLY/WTA representation and compresses analytic state rather than proposing RanPAC's classifier. |
| `goswami2023fecam` | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/15294ba2dcfb4521274f7aa1c26f4dd4-Abstract-Conference.html) | Covariance-aware class modeling with heterogeneous distributions. | SRQ-FLY stores global code statistics; it neither uses Mahalanobis inference nor class Gaussian replay. |
| `zhuang2024foal` | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/48ffa38c13078d6ce26b328e7f373243-Abstract-Conference.html) | Forward-only online analytic learning with recursive least squares and a frozen encoder. | SRQ-FLY's novelty target is compressed square-root state for a wide WTA representation. |
| `zhuang2024gacl` | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9713d53ee4f31781304b1ca43266f8d1-Abstract-Conference.html) | Generalized analytic CL with exposed/unexposed class decomposition. | SRQ-FLY currently studies disjoint class-incremental task streams, not generalized mixed-class streams. |
| `rypesc2024adagauss` | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/73ba81c7b25134a559c8a9c39ec1a4c3-Abstract-Conference.html) | Adaptive class covariances and pseudo-feature replay for exemplar-free CIL with representation updates. | SRQ-FLY freezes the backbone and generates no pseudo-samples. |
| `he2024real` | [arXiv 2024](https://arxiv.org/abs/2403.13522) | Representation enhancement for exemplar-free recursive analytic CIL. | SRQ-FLY does not adapt the frozen backbone/feature representation; it targets the quadratic classifier state. |
| `li2025fourbit` | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Memory-Efficient_4-bit_Preconditioned_Stochastic_Optimization_ICCV_2025_paper.html) | Cholesky-factor quantization plus error feedback for Shampoo preconditioners. | SRQ-FLY is int8 streaming Ridge without error feedback; Shampoo convergence results are not reused. |
| `momeni2025anacp` | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/699b19e638a086ed3a6d1710c5aea504-Abstract-Conference.html) | Analytic contrastive projection for continual feature adaptation. | SRQ-FLY intentionally leaves the frozen representation and code semantics unchanged. |
| `gao2025moal` | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Knowledge_Memorization_and_Rumination_for_Pre-trained_Model-based_Class-Incremental_Learning_CVPR_2025_paper.html) | Momentum-based analytic learning for better new-class adaptation with pretrained models. | SRQ-FLY currently makes no representation-adaptation contribution; its target is state compression and numerical structure. |
| `zou2026flycl` | [ICLR 2026](https://openreview.net/forum?id=jNbxjdc745) | Fly-inspired pretrained-model CL aimed at decorrelation and low training cost. | This is the direct representation baseline; SRQ-FLY changes its persistent analytic-state representation, not its backbone or WTA mapping. |

## Local SOHO boundary

The checked-out repository contains a SOHO implementation with OLDA,
ETF/Procrustes alignment, dynamic WTA, and historical-feature re-projection.
No authoritative SOHO publication or final technical report is present in the
repository. Until one is supplied, the manuscript may describe only the
audited local implementation and must not attribute literature priority,
peer-review status, or external results to “SOHO.”
