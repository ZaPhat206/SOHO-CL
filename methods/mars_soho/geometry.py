"""Numerically stable SOHO projection and hard-WTA support geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .statistics import MomentSnapshot


@dataclass(frozen=True)
class ProjectionResult:
    rotation: torch.Tensor
    eigenvalues: torch.Tensor
    discriminative_rank: int


def _align_row_block(new: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    if new.shape != previous.shape:
        raise ValueError("gauge-alignment blocks must have the same shape")
    if new.shape[0] == 0:
        return new
    left, _, right = torch.linalg.svd(previous @ new.T)
    return (left @ right) @ new


def align_projection_gauge(
    rotation: torch.Tensor,
    previous_rotation: torch.Tensor,
    *,
    discriminative_rank: int,
    eigenvalues: torch.Tensor,
    use_etf: bool,
    relative_tolerance: float = 1e-6,
) -> torch.Tensor:
    """Align only mathematically non-identifiable rotation blocks.

    ETF fixes the orientation of the discriminative block using class geometry,
    so only its null complement is aligned. Without ETF, adjacent nearly equal
    generalized eigenvalues define an arbitrary basis and are Procrustes-aligned
    blockwise. No rotation is allowed between unequal-eigenvalue subspaces.
    """
    if rotation.shape != previous_rotation.shape or rotation.ndim != 2:
        raise ValueError("rotation and previous_rotation must share shape (d,D)")
    if not 0 <= discriminative_rank <= rotation.shape[0]:
        raise ValueError("invalid discriminative rank")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    aligned = rotation.clone()
    if not use_etf and discriminative_rank > 1:
        start = 0
        while start < discriminative_rank:
            stop = start + 1
            while stop < discriminative_rank:
                left_value = eigenvalues[stop - 1]
                right_value = eigenvalues[stop]
                scale = torch.maximum(
                    torch.ones((), device=eigenvalues.device, dtype=eigenvalues.dtype),
                    torch.maximum(left_value.abs(), right_value.abs()),
                )
                if bool((left_value - right_value).abs() > relative_tolerance * scale):
                    break
                stop += 1
            if stop - start > 1:
                aligned[start:stop] = _align_row_block(
                    aligned[start:stop], previous_rotation[start:stop]
                )
            start = stop
    if discriminative_rank < rotation.shape[0]:
        aligned[discriminative_rank:] = _align_row_block(
            aligned[discriminative_rank:], previous_rotation[discriminative_rank:]
        )
    return aligned


def _canonicalize_column_signs(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    columns = matrix.clone()
    pivots = columns.abs().argmax(dim=0)
    signs = torch.sign(columns[pivots, torch.arange(columns.shape[1], device=columns.device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return columns * signs.unsqueeze(0)


def compute_soho_rotation(
    snapshot: MomentSnapshot,
    *,
    output_dim: int,
    use_etf: bool,
    within_regularization: float = 1e-4,
) -> ProjectionResult:
    """Compute the SOHO OLDA/ETF basis using solves rather than an inverse."""
    if snapshot.total_count <= 0 or snapshot.num_classes <= 0:
        raise ValueError("at least one observed feature is required")
    if not 0 < output_dim <= snapshot.feature_dim:
        raise ValueError("output_dim must be in [1, feature_dim]")
    if within_regularization <= 0:
        raise ValueError("within_regularization must be positive")
    means = snapshot.means
    centered = means - snapshot.global_mean.unsqueeze(1)
    between = (centered * snapshot.counts.sqrt().unsqueeze(0))
    between_scatter = between @ between.T / snapshot.total_count
    within = snapshot.within_scatter / snapshot.total_count
    eye = torch.eye(snapshot.feature_dim, device=within.device, dtype=within.dtype)
    factor = torch.linalg.cholesky((within + within.T) * 0.5 + within_regularization * eye)
    left = torch.linalg.solve_triangular(factor, between_scatter, upper=False)
    whitened = torch.linalg.solve_triangular(factor, left.T, upper=False).T
    whitened = (whitened + whitened.T) * 0.5
    eigenvalues, whitened_vectors = torch.linalg.eigh(whitened)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    whitened_vectors = whitened_vectors[:, order]
    vectors = torch.linalg.solve_triangular(
        factor.T, whitened_vectors, upper=True
    )
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=0)
    discriminative_rank = min(snapshot.num_classes - 1, snapshot.feature_dim)
    if discriminative_rank:
        discriminative, _ = torch.linalg.qr(
            vectors[:, :discriminative_rank], mode="reduced"
        )
        discriminative = _canonicalize_column_signs(discriminative)
    else:
        discriminative = torch.empty(
            (snapshot.feature_dim, 0), device=within.device, dtype=within.dtype
        )
    if use_etf and snapshot.num_classes > 1 and discriminative_rank == snapshot.num_classes - 1:
        centered_identity = eye[: snapshot.num_classes, : snapshot.num_classes]
        centered_identity = centered_identity - torch.ones_like(centered_identity) / snapshot.num_classes
        etf_vectors, _, _ = torch.linalg.svd(centered_identity)
        etf = (snapshot.num_classes / (snapshot.num_classes - 1)) ** 0.5 * etf_vectors[:, :discriminative_rank].T
        class_geometry = discriminative.T @ centered
        class_geometry = torch.nn.functional.normalize(class_geometry, p=2, dim=0)
        left_proc, _, right_proc = torch.linalg.svd(class_geometry @ etf.T)
        discriminative = discriminative @ (left_proc @ right_proc)
    if discriminative_rank:
        complete, _ = torch.linalg.qr(discriminative, mode="complete")
    else:
        complete = eye
    full_basis = torch.cat(
        (discriminative, complete[:, discriminative_rank:]), dim=1
    )
    rotation = full_basis[:, :output_dim].T.contiguous()
    return ProjectionResult(rotation, eigenvalues, discriminative_rank)


def topk_margin(expanded: torch.Tensor, k: int) -> torch.Tensor:
    if expanded.ndim != 2:
        raise ValueError("expanded activations must have shape (N, M)")
    if not 0 < k < expanded.shape[1]:
        raise ValueError("k must lie in [1, M-1]")
    boundary = torch.topk(expanded, k + 1, dim=1, largest=True).values
    return boundary[:, k - 1] - boundary[:, k]


def certified_stable_support(
    old_expanded: torch.Tensor, new_expanded: torch.Tensor, k: int
) -> torch.Tensor:
    if old_expanded.shape != new_expanded.shape:
        raise ValueError("old and new activations must have the same shape")
    perturbation = (new_expanded - old_expanded).abs().amax(dim=1)
    return topk_margin(old_expanded, k) > 2 * perturbation


def topk_support_turnover(
    old_expanded: torch.Tensor, new_expanded: torch.Tensor, k: int
) -> torch.Tensor:
    """Return the fraction of old Top-K indices replaced for each row.

    Unlike the sufficient (and deliberately conservative) support certificate,
    this is an exact continuous observation of the two realized supports.  It
    is a diagnostic of map drift, not a certificate for unseen samples.
    """
    if old_expanded.shape != new_expanded.shape or old_expanded.ndim != 2:
        raise ValueError("old and new activations must share shape (N, M)")
    if not 0 < k < old_expanded.shape[1]:
        raise ValueError("k must lie in [1, M-1]")
    old_indices = torch.topk(old_expanded, k, dim=1, largest=True).indices
    new_indices = torch.topk(new_expanded, k, dim=1, largest=True).indices
    old_membership = torch.zeros_like(old_expanded, dtype=torch.bool)
    old_membership.scatter_(1, old_indices, True)
    retained = old_membership.gather(1, new_indices).sum(dim=1)
    return 1 - retained.to(old_expanded.dtype) / k
