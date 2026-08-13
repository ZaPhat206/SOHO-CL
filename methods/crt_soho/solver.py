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
