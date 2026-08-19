"""Analytic solvers for low-rank plus exact-diagonal FLY statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RidgeSolution:
    weights: torch.Tensor
    relative_residual: float
    active_rank: int


def _validate(
    U: torch.Tensor,
    s: torch.Tensor,
    diagonal: torch.Tensor,
    Q: torch.Tensor,
) -> None:
    if U.ndim != 2 or s.ndim != 1 or diagonal.ndim != 1 or Q.ndim != 2:
        raise ValueError("invalid solver tensor rank")
    if U.shape[0] != len(diagonal) or Q.shape[0] != len(diagonal):
        raise ValueError("solver feature dimensions disagree")
    if U.shape[1] != len(s):
        raise ValueError("U and s dimensions disagree")
    if bool((s < 0).any() or (diagonal < 0).any()):
        raise ValueError("singular values and diagonal must be non-negative")
    if not bool(
        torch.isfinite(U).all()
        and torch.isfinite(s).all()
        and torch.isfinite(diagonal).all()
        and torch.isfinite(Q).all()
    ):
        raise ValueError("solver inputs contain NaN or Inf")


def low_rank_diagonal(U: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    if U.ndim != 2 or s.ndim != 1 or U.shape[1] != len(s):
        raise ValueError("invalid low-rank factors")
    return (U.square() * s.square().unsqueeze(0)).sum(dim=1)


def diagonal_tail(
    exact_diagonal: torch.Tensor,
    U: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    if exact_diagonal.ndim != 1 or U.shape[0] != len(exact_diagonal):
        raise ValueError("diagonal and factor dimensions disagree")
    return (exact_diagonal - low_rank_diagonal(U, s)).clamp_min(0)


def approximate_gram(
    U: torch.Tensor,
    s: torch.Tensor,
    tail: torch.Tensor,
) -> torch.Tensor:
    if U.shape[0] != len(tail):
        raise ValueError("tail dimension mismatch")
    return (U * s.unsqueeze(0)) @ (U * s.unsqueeze(0)).T + torch.diag(tail)


def _relative_residual(
    weights: torch.Tensor,
    U: torch.Tensor,
    s: torch.Tensor,
    tail: torch.Tensor,
    Q: torch.Tensor,
    ridge_lambda: float,
) -> float:
    applied = U @ (s.square().unsqueeze(1) * (U.T @ weights))
    applied += (tail + ridge_lambda).unsqueeze(1) * weights
    numerator = torch.linalg.vector_norm(applied - Q)
    denominator = max(float(torch.linalg.vector_norm(Q).item()), 1.0)
    return float(numerator.item()) / denominator


def solve_tail_ridge(
    U: torch.Tensor,
    s: torch.Tensor,
    tail: torch.Tensor,
    Q: torch.Tensor,
    ridge_lambda: float,
    *,
    solve_dtype: torch.dtype | None = None,
) -> RidgeSolution:
    """Solve low-rank-plus-diagonal Ridge using a stable Woodbury system."""
    _validate(U, s, tail, Q)
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")
    working_dtype = U.dtype if solve_dtype is None else solve_dtype
    if working_dtype not in {torch.float32, torch.float64}:
        raise ValueError("solve_dtype must be float32 or float64")
    work_U = U.to(working_dtype)
    work_s = s.to(working_dtype)
    work_tail = tail.to(working_dtype)
    work_Q = Q.to(working_dtype)
    threshold = torch.finfo(working_dtype).eps * max(
        float(work_s.max().item()) if len(work_s) else 1.0, 1.0
    ) * max(work_U.shape)
    active = work_s > threshold
    active_U, active_s = work_U[:, active], work_s[active]
    diagonal = work_tail + ridge_lambda
    base = work_Q / diagonal.unsqueeze(1)
    if active_s.numel():
        scaled = active_U * active_s.unsqueeze(0)
        diagonal_scaled = scaled / diagonal.unsqueeze(1)
        middle = torch.eye(
            len(active_s), device=work_U.device, dtype=working_dtype
        ) + scaled.T @ diagonal_scaled
        factor = torch.linalg.cholesky((middle + middle.T) * 0.5)
        right = scaled.T @ base
        correction = torch.cholesky_solve(right, factor)
        weights = base - diagonal_scaled @ correction
    else:
        weights = base
    residual = _relative_residual(
        weights, active_U, active_s, work_tail, work_Q, ridge_lambda
    )
    return RidgeSolution(weights, residual, int(active_s.numel()))


def solve_diagonal_ridge(
    exact_diagonal: torch.Tensor,
    Q: torch.Tensor,
    ridge_lambda: float,
    *,
    solve_dtype: torch.dtype | None = None,
) -> RidgeSolution:
    empty_U = Q.new_empty((Q.shape[0], 0))
    empty_s = Q.new_empty((0,))
    return solve_tail_ridge(
        empty_U,
        empty_s,
        exact_diagonal,
        Q,
        ridge_lambda,
        solve_dtype=solve_dtype,
    )


def solve_truncated_svd_ridge(
    U: torch.Tensor,
    s: torch.Tensor,
    Q: torch.Tensor,
    ridge_lambda: float,
    *,
    solve_dtype: torch.dtype | None = None,
) -> RidgeSolution:
    """Projected TSVD-Ridge control that discards all orthogonal directions."""
    zeros = Q.new_zeros(Q.shape[0])
    _validate(U, s, zeros, Q)
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")
    working_dtype = U.dtype if solve_dtype is None else solve_dtype
    if working_dtype not in {torch.float32, torch.float64}:
        raise ValueError("solve_dtype must be float32 or float64")
    work_U = U.to(working_dtype)
    work_s = s.to(working_dtype)
    work_Q = Q.to(working_dtype)
    if not len(work_s):
        return RidgeSolution(torch.zeros_like(work_Q), 0.0, 0)
    coefficients = (work_U.T @ work_Q) / (
        work_s.square() + ridge_lambda
    ).unsqueeze(1)
    weights = work_U @ coefficients
    projected_residual = (
        (work_s.square() + ridge_lambda).unsqueeze(1) * (work_U.T @ weights)
        - work_U.T @ work_Q
    )
    denominator = max(
        float(torch.linalg.vector_norm(work_U.T @ work_Q).item()), 1.0
    )
    relative = float(torch.linalg.vector_norm(projected_residual).item()) / denominator
    return RidgeSolution(weights, relative, len(work_s))
