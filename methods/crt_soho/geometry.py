"""Anchor confusion and residual direction controls."""

from __future__ import annotations

import torch

from .statistics import DualViewStatistics


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.T) * 0.5


def solve_spd(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    factor = torch.linalg.cholesky(symmetrize(matrix))
    return torch.cholesky_solve(rhs, factor)


def anchor_weights(statistics: DualViewStatistics, ridge: float) -> torch.Tensor:
    if ridge <= 0:
        raise ValueError("anchor ridge must be positive")
    eye = torch.eye(statistics.anchor_dim, dtype=statistics.dtype, device=statistics.device)
    return solve_spd(statistics.G_pp + ridge * eye, statistics.Q_p)


def relative_margin_affinity(statistics: DualViewStatistics, weights: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric relative confusion from expected anchor-classifier margins."""
    if temperature <= 0:
        raise ValueError("confusion temperature must be positive")
    classes = statistics.num_classes
    if classes <= 1:
        return torch.zeros((classes, classes), device=statistics.device, dtype=statistics.dtype)
    means = statistics.Q_p / statistics.counts.clamp_min(1).unsqueeze(0)
    scores = means.T @ weights
    margins = torch.diag(scores).unsqueeze(1) - scores
    logits = -margins / temperature
    logits.fill_diagonal_(-torch.inf)
    directed = torch.softmax(logits, dim=1)
    affinity = symmetrize(directed)
    affinity.fill_diagonal_(0)
    return affinity


def shuffled_affinity(affinity: torch.Tensor, seed: int) -> torch.Tensor:
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


def raw_scatter(statistics: DualViewStatistics) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = statistics.Q_x / statistics.counts.clamp_min(1).unsqueeze(0)
    within = statistics.G_xx - (means * statistics.counts.unsqueeze(0)) @ means.T
    global_mean = (means * statistics.counts.unsqueeze(0)).sum(1) / statistics.counts.sum().clamp_min(1)
    centered = means - global_mean.unsqueeze(1)
    between = (centered * statistics.counts.unsqueeze(0)) @ centered.T
    return symmetrize(within), symmetrize(between), means


def pairwise_between(means: torch.Tensor, counts: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
    differences = means.T[:, None, :] - means.T[None, :, :]
    pair_mass = counts[:, None] * counts[None, :] / (counts[:, None] + counts[None, :]).clamp_min(1)
    return symmetrize(0.5 * torch.einsum("ij,ijd,ije->de", affinity * pair_mass, differences, differences))


def _canonicalize_columns(matrix: torch.Tensor) -> torch.Tensor:
    result = matrix.clone()
    for column in range(result.shape[1]):
        pivot = torch.argmax(result[:, column].abs())
        if result[pivot, column] < 0:
            result[:, column].mul_(-1)
    return result


def fisher_directions(within: torch.Tensor, between: torch.Tensor, total_count: int, classes: int, rank: int, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a Euclidean-orthonormal basis of the top Fisher subspace."""
    dimension = within.shape[0]
    if rank <= 0 or rank > dimension:
        raise ValueError(f"rank must be in [1, {dimension}]")
    if epsilon <= 0:
        raise ValueError("scatter epsilon must be positive")
    covariance = symmetrize(within / max(float(total_count - classes), 1.0))
    covariance += epsilon * torch.eye(dimension, dtype=within.dtype, device=within.device)
    lower = torch.linalg.cholesky(covariance)
    left = torch.linalg.solve_triangular(lower, symmetrize(between / max(float(total_count), 1.0)), upper=False)
    whitened = symmetrize(torch.linalg.solve_triangular(lower, left.T, upper=False).T)
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[order].clamp_min(0), eigenvectors[:, order]
    generalized = torch.linalg.solve_triangular(lower.T, eigenvectors[:, :rank], upper=True)
    # Direction scaling is isolated from residual-ridge tuning: only the
    # selected subspace differs between structured controls.
    orthonormal, _ = torch.linalg.qr(generalized, mode="reduced")
    return _canonicalize_columns(orthonormal), eigenvalues


def random_directions(dimension: int, rank: int, seed: int, dtype, device) -> torch.Tensor:
    if rank <= 0 or rank > dimension:
        raise ValueError(f"rank must be in [1, {dimension}]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn((dimension, rank), generator=generator, dtype=dtype).to(device)
    directions, _ = torch.linalg.qr(raw, mode="reduced")
    return _canonicalize_columns(directions)
