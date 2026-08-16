"""Deterministic analytic solvers for Two-Way Analytic FLY."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .statistics import TWAStatistics


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.T) * 0.5


def _factor(system: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cholesky(_symmetrize(system))


def _solve(factor: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.cholesky_solve(rhs, factor)


def _systems(statistics: TWAStatistics, rho: float, lambda_x: float, lambda_z: float):
    if rho < 0 or lambda_x <= 0 or lambda_z <= 0:
        raise ValueError("rho must be non-negative and both Ridge coefficients positive")
    ix = torch.eye(statistics.raw_dim, device=statistics.device, dtype=statistics.dtype)
    iz = torch.eye(statistics.fly_dim, device=statistics.device, dtype=statistics.dtype)
    ax = (1.0 + rho) * _symmetrize(statistics.G_xx) + lambda_x * ix
    az = (1.0 + rho) * _symmetrize(statistics.G_zz) + lambda_z * iz
    return ax, az


def block_relative_residual(
    statistics: TWAStatistics,
    raw_weights: torch.Tensor,
    fly_weights: torch.Tensor,
    rho: float,
    lambda_x: float,
    lambda_z: float,
    cross: torch.Tensor | None = None,
) -> float:
    if rho < 0 or lambda_x <= 0 or lambda_z <= 0:
        raise ValueError("rho must be non-negative and both Ridge coefficients positive")
    r = statistics.R_xz if cross is None else cross
    residual_x = (
        (1.0 + rho) * (statistics.G_xx @ raw_weights)
        + lambda_x * raw_weights
        - rho * r @ fly_weights
        - statistics.Q_x
    )
    residual_z = (
        (1.0 + rho) * (statistics.G_zz @ fly_weights)
        + lambda_z * fly_weights
        - rho * r.T @ raw_weights
        - statistics.Q_z
    )
    numerator = torch.sqrt(residual_x.square().sum() + residual_z.square().sum())
    denominator = torch.sqrt(statistics.Q_x.square().sum() + statistics.Q_z.square().sum()).clamp_min(
        torch.finfo(statistics.dtype).eps
    )
    return float((numerator / denominator).item())


def objective_without_constant(
    statistics: TWAStatistics,
    raw_weights: torch.Tensor,
    fly_weights: torch.Tensor,
    rho: float,
    lambda_x: float,
    lambda_z: float,
    cross: torch.Tensor | None = None,
) -> float:
    r = statistics.R_xz if cross is None else cross
    raw_fit = torch.sum(raw_weights * (statistics.G_xx @ raw_weights)) - 2 * torch.sum(raw_weights * statistics.Q_x)
    fly_fit = torch.sum(fly_weights * (statistics.G_zz @ fly_weights)) - 2 * torch.sum(fly_weights * statistics.Q_z)
    agreement = (
        torch.sum(raw_weights * (statistics.G_xx @ raw_weights))
        + torch.sum(fly_weights * (statistics.G_zz @ fly_weights))
        - 2 * torch.sum(raw_weights * (r @ fly_weights))
    )
    penalty = lambda_x * raw_weights.square().sum() + lambda_z * fly_weights.square().sum()
    return float((raw_fit + fly_fit + rho * agreement + penalty).item())


@dataclass(frozen=True)
class TWASolution:
    raw_weights: torch.Tensor
    fly_weights: torch.Tensor
    relative_residual: float
    iterations: int
    objective_history: tuple[float, ...]


@dataclass(frozen=True)
class CoupledFactors:
    raw_factor: torch.Tensor
    fly_factor: torch.Tensor


def factor_coupled_systems(
    statistics: TWAStatistics, rho: float, lambda_x: float, lambda_z: float
) -> CoupledFactors:
    """Factor the two diagonal blocks once for a family of cross controls."""
    ax, az = _systems(statistics, rho, lambda_x, lambda_z)
    return CoupledFactors(_factor(ax), _factor(az))


def solve_one_way(
    statistics: TWAStatistics,
    rho: float,
    lambda_x: float,
    lambda_z: float,
    *,
    cross: torch.Tensor | None = None,
    raw_teacher: torch.Tensor | None = None,
    fly_factor: torch.Tensor | None = None,
) -> TWASolution:
    if rho < 0:
        raise ValueError("rho must be non-negative")
    r = statistics.R_xz if cross is None else cross
    if r.shape != statistics.R_xz.shape:
        raise ValueError("cross statistic has invalid shape")
    if raw_teacher is None:
        ix = torch.eye(statistics.raw_dim, device=statistics.device, dtype=statistics.dtype)
        raw_factor = _factor(statistics.G_xx + lambda_x * ix)
        raw_weights = _solve(raw_factor, statistics.Q_x)
    else:
        raw_weights = raw_teacher
        if raw_weights.shape != statistics.Q_x.shape:
            raise ValueError("raw_teacher has invalid shape")
    if fly_factor is None:
        _, az = _systems(statistics, rho, lambda_x, lambda_z)
        fz = _factor(az)
    else:
        fz = fly_factor
    fly_weights = _solve(fz, statistics.Q_z + rho * r.T @ raw_weights)
    ix = torch.eye(statistics.raw_dim, device=statistics.device, dtype=statistics.dtype)
    raw_residual = (statistics.G_xx + lambda_x * ix) @ raw_weights - statistics.Q_x
    fly_rhs = statistics.Q_z + rho * r.T @ raw_weights
    fly_residual = (
        (1.0 + rho) * (statistics.G_zz @ fly_weights)
        + lambda_z * fly_weights
        - fly_rhs
    )
    numerator = torch.sqrt(raw_residual.square().sum() + fly_residual.square().sum())
    denominator = torch.sqrt(statistics.Q_x.square().sum() + fly_rhs.square().sum()).clamp_min(
        torch.finfo(statistics.dtype).eps
    )
    residual = float((numerator / denominator).item())
    objective = objective_without_constant(
        statistics, raw_weights, fly_weights, rho, lambda_x, lambda_z, cross=r
    )
    return TWASolution(raw_weights, fly_weights, residual, 1, (objective,))


def solve_symmetric(
    statistics: TWAStatistics,
    rho: float,
    lambda_x: float,
    lambda_z: float,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    cross: torch.Tensor | None = None,
    factors: CoupledFactors | None = None,
) -> TWASolution:
    if tolerance <= 0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")
    r = statistics.R_xz if cross is None else cross
    if r.shape != statistics.R_xz.shape:
        raise ValueError("cross statistic has invalid shape")
    if factors is None:
        factors = factor_coupled_systems(statistics, rho, lambda_x, lambda_z)
    fx, fz = factors.raw_factor, factors.fly_factor
    # The rho=0 branch is the mandatory exact matched-FLY identity.
    raw_weights = _solve(fx, statistics.Q_x)
    fly_weights = _solve(fz, statistics.Q_z)
    history = [objective_without_constant(
        statistics, raw_weights, fly_weights, rho, lambda_x, lambda_z, cross=r
    )]
    residual = block_relative_residual(
        statistics, raw_weights, fly_weights, rho, lambda_x, lambda_z, cross=r
    )
    if residual <= tolerance:
        return TWASolution(raw_weights, fly_weights, residual, 0, tuple(history))
    for iteration in range(1, max_iterations + 1):
        raw_weights = _solve(fx, statistics.Q_x + rho * r @ fly_weights)
        fly_weights = _solve(fz, statistics.Q_z + rho * r.T @ raw_weights)
        objective = objective_without_constant(
            statistics, raw_weights, fly_weights, rho, lambda_x, lambda_z, cross=r
        )
        previous = history[-1]
        slack = 64 * torch.finfo(statistics.dtype).eps * max(1.0, abs(previous))
        if objective > previous + slack:
            raise RuntimeError("alternating objective increased beyond numerical tolerance")
        history.append(objective)
        residual = block_relative_residual(
            statistics, raw_weights, fly_weights, rho, lambda_x, lambda_z, cross=r
        )
        if not torch.isfinite(torch.tensor(residual)):
            raise RuntimeError("TWA solver produced a non-finite residual")
        if residual <= tolerance:
            return TWASolution(raw_weights, fly_weights, residual, iteration, tuple(history))
    raise RuntimeError(
        f"TWA solver did not reach relative residual {tolerance:g} in {max_iterations} iterations; "
        f"observed {residual:.6g}"
    )
