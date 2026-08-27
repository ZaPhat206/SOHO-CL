"""Class geometry and exact moment transport for MT-SOHO."""

from __future__ import annotations

import torch


def solve_spd(matrix: torch.Tensor, right_hand_side: torch.Tensor) -> torch.Tensor:
    """Solve a symmetric positive-definite system without an explicit inverse."""
    symmetric = (matrix + matrix.T) * 0.5
    factor = torch.linalg.cholesky(symmetric)
    return torch.cholesky_solve(right_hand_side, factor)


def class_geometry(
    *,
    raw_gram: torch.Tensor,
    raw_cross: torch.Tensor,
    counts: torch.Tensor,
    requested_rank: int,
    shrinkage: float,
    epsilon: float,
    whiten: bool,
) -> tuple[torch.Tensor, dict]:
    """Build deterministic class targets from bounded raw-feature moments.

    ``raw_cross[:, c]`` is the sum of raw features in class ``c``.  Therefore
    the pooled within-class scatter is exactly

        X^T X - sum_c n_c mu_c mu_c^T.

    The returned target matrix has shape ``(C_seen, r)``.  Its rows are unit
    norm so target scale cannot silently act as another hyperparameter.
    """
    if requested_rank <= 0:
        raise ValueError("requested_rank must be positive")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    classes = int(counts.numel())
    if classes < 2 or bool((counts <= 0).any()):
        raise ValueError("class geometry requires at least two non-empty classes")

    means = raw_cross / counts.unsqueeze(0)
    total = counts.sum()
    global_mean = raw_cross.sum(1, keepdim=True) / total
    centered = means - global_mean
    within = raw_gram - (raw_cross / counts.unsqueeze(0)) @ raw_cross.T
    within = (within + within.T) * 0.5
    denominator = max(int(total.item()) - classes, 1)
    covariance = within / denominator
    trace_scale = covariance.diagonal().mean().clamp_min(epsilon)
    regularized = (
        (1.0 - shrinkage) * covariance
        + (shrinkage * trace_scale + epsilon)
        * torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    )

    if whiten:
        eigenvalues, eigenvectors = torch.linalg.eigh(regularized)
        floor = epsilon * trace_scale.clamp_min(1.0)
        inverse_root = (eigenvectors * eigenvalues.clamp_min(floor).rsqrt()) @ eigenvectors.T
        geometry = inverse_root @ centered
    else:
        eigenvalues = torch.linalg.eigvalsh(regularized)
        geometry = centered

    effective_rank = min(int(requested_rank), classes - 1, geometry.shape[0])
    left, singular_values, _ = torch.linalg.svd(geometry, full_matrices=False)
    basis = left[:, :effective_rank]
    targets = (basis.T @ geometry).T
    targets = torch.nn.functional.normalize(targets, dim=1, eps=epsilon)
    diagnostics = {
        "effective_rank": effective_rank,
        "within_trace": float(within.diagonal().sum().item()),
        "covariance_condition": float(
            eigenvalues.max().div(eigenvalues.min().clamp_min(epsilon)).item()
        ),
        "target_singular_values": singular_values[:effective_rank].detach().clone(),
    }
    return targets, diagnostics


def shuffled_targets(targets: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(targets.shape[0], generator=generator).to(targets.device)
    return targets[permutation]


def transport_moments(
    gram: torch.Tensor,
    cross: torch.Tensor,
    transport: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport fixed-anchor sufficient statistics exactly."""
    return transport.T @ gram @ transport, transport.T @ cross
