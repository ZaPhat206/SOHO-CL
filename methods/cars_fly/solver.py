"""Certified adaptive-rank conditional correction for CARS-FLY."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from methods.crt_soho.geometry import solve_spd, symmetrize
from methods.crt_soho.solver import reconstruct_residual_statistics
from methods.crt_soho.statistics import DualViewStatistics


@dataclass(frozen=True)
class ConditionalCorrection:
    directions: torch.Tensor
    singular_values: torch.Tensor
    effective_rank: int
    captured_energy: float
    tail_energy: float
    total_energy: float
    retained_fraction: float
    threshold_reached: bool


def adaptive_conditional_directions(
    statistics: DualViewStatistics,
    *,
    complement_ridge: float,
    anchor_ridge: float,
    residual_ridge: float,
    energy_threshold: float,
    max_rank: int,
    min_rank: int = 1,
    minimum_objective_gain: float = 0.0,
) -> ConditionalCorrection:
    """Select the smallest bounded rank reaching the Schur-energy threshold.

    The singular spectrum is that of the label-predictive correction left after
    eliminating the fixed nonlinear anchor from the regularized block system.
    No sample rows are needed: every matrix is reconstructed from aggregate
    dual-view sufficient statistics.
    """
    if min(complement_ridge, anchor_ridge, residual_ridge) <= 0:
        raise ValueError("all Ridge coefficients must be positive")
    if not 0 < energy_threshold <= 1:
        raise ValueError("energy_threshold must be in (0, 1]")
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    if min_rank <= 0 or min_rank > max_rank:
        raise ValueError("min_rank must be in [1, max_rank]")
    if minimum_objective_gain < 0:
        raise ValueError("minimum_objective_gain must be non-negative")

    identity = torch.eye(
        statistics.raw_dim, dtype=statistics.dtype, device=statistics.device
    )
    full = reconstruct_residual_statistics(
        statistics, identity, complement_ridge, residualize=True
    )
    anchor_identity = torch.eye(
        statistics.anchor_dim,
        dtype=statistics.dtype,
        device=statistics.device,
    )
    anchor_system = symmetrize(statistics.G_pp) + anchor_ridge * anchor_identity
    solved_cross = solve_spd(anchor_system, full.G_pr)
    solved_targets = solve_spd(anchor_system, statistics.Q_p)
    schur = symmetrize(
        full.G_rr + residual_ridge * identity - full.G_pr.T @ solved_cross
    )
    targets = full.Q_r - full.G_pr.T @ solved_targets
    lower = torch.linalg.cholesky(schur)
    whitened_targets = torch.linalg.solve_triangular(
        lower, targets, upper=False
    )
    left, singular_values, _ = torch.linalg.svd(
        whitened_targets, full_matrices=False
    )
    if not bool(torch.isfinite(singular_values).all()):
        raise RuntimeError("conditional spectrum contains NaN or Inf")

    energies = singular_values.square()
    total_tensor = energies.sum()
    total_energy = float(total_tensor.item())
    available_rank = min(int(max_rank), int(left.shape[1]), statistics.raw_dim)
    selected_rank = 0
    if total_energy > minimum_objective_gain and available_rank > 0:
        target = energy_threshold * total_tensor
        selected_rank = int(
            torch.searchsorted(energies.cumsum(0), target).item()
        ) + 1
        selected_rank = min(max(selected_rank, min_rank), available_rank)

    if selected_rank:
        generalized = torch.linalg.solve_triangular(
            lower.T, left[:, :selected_rank], upper=True
        )
        directions, _ = torch.linalg.qr(generalized, mode="reduced")
        # Resolve QR sign ambiguity so checkpoints rebuild deterministically.
        for column in range(directions.shape[1]):
            pivot = torch.argmax(directions[:, column].abs())
            if directions[pivot, column] < 0:
                directions[:, column].mul_(-1)
    else:
        directions = torch.empty(
            (statistics.raw_dim, 0),
            dtype=statistics.dtype,
            device=statistics.device,
        )

    captured_energy = float(energies[:selected_rank].sum().item())
    tail_energy = max(total_energy - captured_energy, 0.0)
    retained_fraction = (
        captured_energy / total_energy if total_energy > 0 else 1.0
    )
    tolerance = 16 * torch.finfo(statistics.dtype).eps
    return ConditionalCorrection(
        directions=directions,
        singular_values=singular_values,
        effective_rank=selected_rank,
        captured_energy=captured_energy,
        tail_energy=tail_energy,
        total_energy=total_energy,
        retained_fraction=retained_fraction,
        threshold_reached=(
            total_energy <= minimum_objective_gain
            or retained_fraction + tolerance >= energy_threshold
        ),
    )
