"""Deterministic Fisher geometry reconstructed from sufficient statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .statistics import FixedFeatureStatistics


@dataclass(frozen=True)
class FisherGeometry:
    within_scatter: torch.Tensor
    between_scatter: torch.Tensor
    affinity: torch.Tensor | None
    eigenvalues: torch.Tensor
    gains: torch.Tensor
    effective_rank: int


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.T) * 0.5


def scatter_matrices(statistics: FixedFeatureStatistics) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact within/between scatter and class means from G, Q, counts."""
    if statistics.num_classes == 0:
        raise RuntimeError("at least one class is required")
    means = statistics.means()  # (D, C)
    counts = statistics.counts
    within = statistics.G - (means * counts.unsqueeze(0)) @ means.T
    global_mean = (means * counts.unsqueeze(0)).sum(dim=1) / counts.sum().clamp_min(1)
    centered = means - global_mean.unsqueeze(1)
    between = (centered * counts.unsqueeze(0)) @ centered.T
    return _symmetrize(within), _symmetrize(between), means


def raw_ridge_weights(statistics: FixedFeatureStatistics, ridge_lambda: float) -> torch.Tensor:
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")
    eye = torch.eye(statistics.feature_dim, dtype=statistics.dtype, device=statistics.device)
    return torch.linalg.solve(statistics.G + ridge_lambda * eye, statistics.Q)


def analytic_confusion_affinity(
    means: torch.Tensor,
    within_scatter: torch.Tensor,
    counts: torch.Tensor,
    raw_weights: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Estimate symmetric pairwise error affinity from pooled Gaussian margins.

    `a[i,j]` is the mean of the analytic probabilities of mistaking class i for
    j and class j for i under a shared pooled covariance approximation.
    """
    classes = means.shape[1]
    if classes <= 1:
        return torch.zeros((classes, classes), dtype=means.dtype, device=means.device)
    denominator = max(float(counts.sum().item() - classes), 1.0)
    covariance = _symmetrize(within_scatter / denominator)
    covariance = covariance + epsilon * torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device)
    # delta[i,j] is w_i - w_j, and margin[i,j] is its mean under class i.
    delta = raw_weights.T[:, None, :] - raw_weights.T[None, :, :]
    margins = torch.einsum("di,ijd->ij", means, delta)
    variances = torch.einsum("ijd,de,ije->ij", delta, covariance, delta).clamp_min(epsilon)
    probabilities = torch.special.ndtr(-margins / variances.sqrt())
    affinity = _symmetrize(torch.nan_to_num((probabilities + probabilities.T) * 0.5, nan=0.0, posinf=0.0, neginf=0.0))
    affinity.fill_diagonal_(0)
    return affinity


def confusion_between_scatter(means: torch.Tensor, counts: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
    """Pairwise Fisher scatter, weighted most strongly for confusable classes."""
    if affinity.shape != (means.shape[1], means.shape[1]):
        raise ValueError("affinity shape must be (C, C)")
    differences = means.T[:, None, :] - means.T[None, :, :]
    pair_mass = counts[:, None] * counts[None, :] / (counts[:, None] + counts[None, :]).clamp_min(1)
    weights = affinity * pair_mass
    # Each unordered pair occurs twice in the full matrix representation.
    return _symmetrize(0.5 * torch.einsum("ij,ijd,ije->de", weights, differences, differences))


def shuffled_affinity(affinity: torch.Tensor, seed: int) -> torch.Tensor:
    """Deterministic semantic control: preserve edge weights, break class pairs."""
    classes = affinity.shape[0]
    result = torch.zeros_like(affinity)
    if classes <= 1:
        return result
    rows, columns = torch.triu_indices(classes, classes, offset=1, device=affinity.device)
    values = affinity[rows, columns]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(values.numel(), generator=generator).to(affinity.device)
    result[rows, columns] = values[order]
    result[columns, rows] = values[order]
    return result


def fisher_transport(
    within_scatter: torch.Tensor,
    between_scatter: torch.Tensor,
    total_count: int,
    scatter_epsilon: float,
    mode: str,
    requested_rank: int | None = None,
    kappa: float = 1.0,
    delta: float = 0.1,
) -> tuple[torch.Tensor, FisherGeometry]:
    """Build A where row features transform as Z=X@A without matrix inversion."""
    if mode not in {"hard", "soft"}:
        raise ValueError("mode must be 'hard' or 'soft'")
    if scatter_epsilon <= 0 or kappa <= 0 or not 0 < delta <= 1:
        raise ValueError("scatter_epsilon and kappa must be positive; delta must be in (0, 1]")
    dimension = within_scatter.shape[0]
    if within_scatter.shape != (dimension, dimension) or between_scatter.shape != (dimension, dimension):
        raise ValueError("scatter matrices must be square with matching shape")
    covariance = _symmetrize(within_scatter / max(float(total_count - 1), 1.0))
    covariance = covariance + scatter_epsilon * torch.eye(dimension, dtype=covariance.dtype, device=covariance.device)
    lower = torch.linalg.cholesky(covariance)
    left = torch.linalg.solve_triangular(lower, _symmetrize(between_scatter / max(float(total_count), 1.0)), upper=False)
    whitened_between = _symmetrize(torch.linalg.solve_triangular(lower, left.T, upper=False).T)
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened_between)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    eigenvectors = eigenvectors.flip(1)

    if mode == "hard":
        if requested_rank is None or requested_rank <= 0:
            raise ValueError("hard Fisher transport requires requested_rank > 0")
        rank = min(int(requested_rank), dimension)
        gains = torch.ones(rank, dtype=within_scatter.dtype, device=within_scatter.device)
    else:
        rank = dimension
        gains = torch.sqrt(delta + (1.0 - delta) * eigenvalues / (eigenvalues + kappa))

    selected = eigenvectors[:, :rank] * gains.unsqueeze(0)
    transport = torch.linalg.solve_triangular(lower.T, selected, upper=True)
    geometry = FisherGeometry(
        within_scatter=within_scatter,
        between_scatter=between_scatter,
        affinity=None,
        eigenvalues=eigenvalues,
        gains=gains,
        effective_rank=rank,
    )
    return transport, geometry
