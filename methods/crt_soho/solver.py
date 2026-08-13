"""Exact residual-statistic reconstruction and block Ridge solve."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import solve_spd, symmetrize
from .statistics import DualViewStatistics


@dataclass(frozen=True)
class ResidualStatistics:
    C: torch.Tensor
    G_pr: torch.Tensor
    G_rr: torch.Tensor
    Q_r: torch.Tensor


@dataclass(frozen=True)
class SchurDirections:
    directions: torch.Tensor
    singular_values: torch.Tensor
    effective_rank: int
    retained_correction_energy: float


def reconstruct_residual_statistics(
    statistics: DualViewStatistics,
    directions: torch.Tensor,
    complement_ridge: float,
    residualize: bool = True,
) -> ResidualStatistics:
    if directions.ndim != 2 or directions.shape[0] != statistics.raw_dim:
        raise ValueError(f"directions must have shape ({statistics.raw_dim}, r)")
    if complement_ridge <= 0:
        raise ValueError("complement_ridge must be positive")
    K = statistics.H_px @ directions
    if residualize:
        eye = torch.eye(statistics.anchor_dim, dtype=statistics.dtype, device=statistics.device)
        C = solve_spd(statistics.G_pp + complement_ridge * eye, K)
    else:
        C = torch.zeros_like(K)
    G_pr = K - statistics.G_pp @ C
    G_rr = (
        directions.T @ statistics.G_xx @ directions
        - K.T @ C
        - C.T @ K
        + C.T @ statistics.G_pp @ C
    )
    Q_r = directions.T @ statistics.Q_x - C.T @ statistics.Q_p
    return ResidualStatistics(C=C, G_pr=G_pr, G_rr=symmetrize(G_rr), Q_r=Q_r)


def solve_block_ridge(
    statistics: DualViewStatistics,
    residual: ResidualStatistics,
    anchor_ridge: float,
    residual_ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the block system via Schur complement and Cholesky factors."""
    if anchor_ridge <= 0 or residual_ridge <= 0:
        raise ValueError("both Ridge coefficients must be positive")
    anchor_eye = torch.eye(statistics.anchor_dim, dtype=statistics.dtype, device=statistics.device)
    residual_eye = torch.eye(residual.G_rr.shape[0], dtype=statistics.dtype, device=statistics.device)
    anchor_system = symmetrize(statistics.G_pp) + anchor_ridge * anchor_eye
    solved_q = solve_spd(anchor_system, statistics.Q_p)
    solved_cross = solve_spd(anchor_system, residual.G_pr)
    schur = symmetrize(residual.G_rr + residual_ridge * residual_eye - residual.G_pr.T @ solved_cross)
    rhs = residual.Q_r - residual.G_pr.T @ solved_q
    residual_weights = solve_spd(schur, rhs)
    anchor_weights = solved_q - solved_cross @ residual_weights
    block = torch.cat((
        torch.cat((anchor_system, residual.G_pr), dim=1),
        torch.cat((residual.G_pr.T, residual.G_rr + residual_ridge * residual_eye), dim=1),
    ), dim=0)
    weights = torch.cat((anchor_weights, residual_weights), dim=0)
    targets = torch.cat((statistics.Q_p, residual.Q_r), dim=0)
    residual_max = float((block @ weights - targets).abs().max().item())
    return anchor_weights, residual_weights, residual_max


def schur_residual_directions(
    statistics: DualViewStatistics,
    complement_ridge: float,
    anchor_ridge: float,
    residual_ridge: float,
    requested_rank: int,
) -> SchurDirections:
    """Select the optimal reduced-rank subspace of the full block correction.

    After analytically eliminating the anchor block, the full residual
    coefficient is ``S^-1 T``. Truncated SVD of ``L^-1 T`` for ``S=L L^T``
    gives its best rank-r approximation in the Schur/Ridge objective. Mapping
    the left singular subspace through ``L^-T`` and Euclidean QR yields an
    orthonormal raw-residual basis compatible with isotropic coefficient Ridge.
    """
    if requested_rank <= 0:
        raise ValueError("requested_rank must be positive")
    if min(complement_ridge, anchor_ridge, residual_ridge) <= 0:
        raise ValueError("all Ridge coefficients must be positive")
    identity = torch.eye(
        statistics.raw_dim, dtype=statistics.dtype, device=statistics.device
    )
    full = reconstruct_residual_statistics(
        statistics, identity, complement_ridge, residualize=True
    )
    anchor_eye = torch.eye(
        statistics.anchor_dim, dtype=statistics.dtype, device=statistics.device
    )
    anchor_system = symmetrize(statistics.G_pp) + anchor_ridge * anchor_eye
    solved_cross = solve_spd(anchor_system, full.G_pr)
    solved_targets = solve_spd(anchor_system, statistics.Q_p)
    schur = symmetrize(
        full.G_rr
        + residual_ridge * identity
        - full.G_pr.T @ solved_cross
    )
    targets = full.Q_r - full.G_pr.T @ solved_targets
    lower = torch.linalg.cholesky(schur)
    whitened_targets = torch.linalg.solve_triangular(lower, targets, upper=False)
    left, singular_values, _ = torch.linalg.svd(whitened_targets, full_matrices=False)
    effective_rank = min(int(requested_rank), left.shape[1])
    generalized = torch.linalg.solve_triangular(
        lower.T, left[:, :effective_rank], upper=True
    )
    directions, _ = torch.linalg.qr(generalized, mode="reduced")
    for column in range(directions.shape[1]):
        pivot = torch.argmax(directions[:, column].abs())
        if directions[pivot, column] < 0:
            directions[:, column].mul_(-1)
    total_energy = float(singular_values.square().sum().item())
    retained = float(singular_values[:effective_rank].square().sum().item())
    retained_fraction = retained / total_energy if total_energy > 0 else 1.0
    return SchurDirections(
        directions=directions,
        singular_values=singular_values,
        effective_rank=effective_rank,
        retained_correction_energy=retained_fraction,
    )
