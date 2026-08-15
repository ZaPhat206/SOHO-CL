"""Compact analytic solver for PPS-SOHO."""

from __future__ import annotations

import torch

from .statistics import ClassProtectedStatistics


def compact_factors(statistics: ClassProtectedStatistics, gamma: float) -> torch.Tensor:
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    sketch = statistics.sketch.B * (gamma ** 0.5)
    if statistics.mode == "global":
        return sketch
    return torch.cat((sketch, statistics.between_factor()), dim=0)


def solve_compact_ridge(
    statistics: ClassProtectedStatistics,
    ridge_lambda: float,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Solve ``(A.T A + lambda I) W = Q`` through Woodbury."""
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")
    if statistics.num_classes == 0:
        raise ValueError("statistics contain no classes")
    factors = compact_factors(statistics, gamma)
    cross = statistics.cross
    small = factors @ factors.T
    small.diagonal().add_(ridge_lambda)
    small = (small + small.T) * 0.5
    lower = torch.linalg.cholesky(small)
    solved = torch.cholesky_solve(factors @ cross, lower)
    weights = (cross - factors.T @ solved) / ridge_lambda
    residual = factors.T @ (factors @ weights) + ridge_lambda * weights - cross
    relative = float(
        residual.abs().max().item() / max(float(cross.abs().max().item()), 1.0)
    )
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("compact Ridge produced non-finite weights")
    return weights, relative
