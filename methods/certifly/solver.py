"""Certified Ridge solve and prediction-stability bounds for CertiFLY."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .quantization import QuantizedSymmetricGram


@dataclass(frozen=True)
class CertifiedRidgeSolution:
    weights: torch.Tensor
    relative_residual: float
    gram_error_bound: float
    relative_classifier_error_bound: float
    absolute_classifier_error_bound: float


def solve_certified_ridge(
    gram: QuantizedSymmetricGram,
    cross: torch.Tensor,
    ridge_lambda: float,
    *,
    solve_dtype: torch.dtype = torch.float32,
) -> CertifiedRidgeSolution:
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")
    if gram.error_bound >= ridge_lambda:
        raise RuntimeError("Gram error certificate does not preserve positive definiteness")
    if cross.ndim != 2 or cross.shape[0] != gram.dimension:
        raise ValueError("cross-statistic shape mismatch")
    if solve_dtype not in {torch.float32, torch.float64}:
        raise ValueError("solve_dtype must be float32 or float64")
    if not bool(torch.isfinite(cross).all()):
        raise ValueError("cross statistic contains NaN or Inf")
    reconstructed = gram.reconstruct(dtype=solve_dtype)
    system = reconstructed + ridge_lambda * torch.eye(
        gram.dimension, device=gram.device, dtype=solve_dtype
    )
    factor, info = torch.linalg.cholesky_ex((system + system.T) * 0.5)
    if int(info.max().item()) != 0:
        raise RuntimeError("certified Ridge system failed Cholesky factorization")
    work_cross = cross.to(device=gram.device, dtype=solve_dtype)
    weights = torch.cholesky_solve(work_cross, factor)
    residual = torch.linalg.vector_norm(system @ weights - work_cross)
    denominator = max(float(torch.linalg.vector_norm(work_cross).item()), 1.0)
    epsilon = gram.error_bound
    # Frobenius norm upper-bounds the spectral norm and is substantially
    # cheaper than a fresh SVD for the tall cross-statistic at every task.
    q_norm = float(torch.linalg.vector_norm(work_cross).item()) if work_cross.numel() else 0.0
    relative_bound = epsilon / (ridge_lambda - epsilon)
    absolute_bound = epsilon * q_norm / (ridge_lambda * (ridge_lambda - epsilon))
    return CertifiedRidgeSolution(
        weights=weights,
        relative_residual=float(residual.item()) / denominator,
        gram_error_bound=epsilon,
        relative_classifier_error_bound=relative_bound,
        absolute_classifier_error_bound=absolute_bound,
    )


def logit_error_bound(
    codes: torch.Tensor, solution: CertifiedRidgeSolution
) -> torch.Tensor:
    if codes.ndim != 2:
        raise ValueError("codes must be a matrix")
    return torch.linalg.vector_norm(codes.to(solution.weights.dtype), dim=1) * solution.absolute_classifier_error_bound


def certified_argmax_mask(
    exact_logits: torch.Tensor,
    per_sample_logit_bound: torch.Tensor,
) -> torch.Tensor:
    if exact_logits.ndim != 2 or per_sample_logit_bound.shape != (len(exact_logits),):
        raise ValueError("invalid logits or bound shape")
    if exact_logits.shape[1] < 2:
        return torch.ones(len(exact_logits), dtype=torch.bool, device=exact_logits.device)
    top_two = exact_logits.topk(2, dim=1).values
    margins = top_two[:, 0] - top_two[:, 1]
    return margins > 2 * per_sample_logit_bound.to(margins)
