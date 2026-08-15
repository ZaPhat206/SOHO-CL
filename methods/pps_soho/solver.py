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
    """Solve ``(A.T A + lambda I) W = Q`` in its compact row space.

    Direct Woodbury evaluation subtracts two nearly equal ``H x C`` tensors
    when lambda is small, which is catastrophically unstable in float32. The
    solution lies in ``span(A.T, Q)``; projection onto an orthonormal basis of
    that space gives an equivalent small SPD system without cancellation.
    """
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")
    if statistics.num_classes == 0:
        raise ValueError("statistics contain no classes")
    factors = compact_factors(statistics, gamma)
    cross = statistics.cross
    # Stored sketches may remain float32 for memory efficiency.  Promote only
    # the compact solve and the resulting classifier: float32 is insufficient
    # for the small lambdas used by the matched FLY protocol.
    solve_dtype = torch.float64 if cross.dtype in {
        torch.float16, torch.bfloat16, torch.float32
    } else cross.dtype
    factors = factors.to(solve_dtype)
    cross = cross.to(solve_dtype)
    source = torch.cat((factors.T, cross), dim=1)
    norms = torch.linalg.vector_norm(source, dim=0)
    nonzero = norms > torch.finfo(source.dtype).tiny
    if not bool(nonzero.any()):
        return torch.zeros_like(cross), 0.0
    normalized = source[:, nonzero] / norms[nonzero]
    left, singular_values, _ = torch.linalg.svd(normalized, full_matrices=False)
    tolerance = (
        torch.finfo(source.dtype).eps
        * max(normalized.shape)
        * singular_values[0]
    )
    basis = left[:, singular_values > tolerance]
    projected_factors = factors @ basis
    small = projected_factors.T @ projected_factors
    small.diagonal().add_(ridge_lambda)
    small = (small + small.T) * 0.5
    lower = torch.linalg.cholesky(small)
    coefficients = torch.cholesky_solve(basis.T @ cross, lower)
    weights = basis @ coefficients
    residual = factors.T @ (factors @ weights) + ridge_lambda * weights - cross
    relative = float(
        residual.abs().max().item() / max(float(cross.abs().max().item()), 1.0)
    )
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("compact Ridge produced non-finite weights")
    return weights, relative
