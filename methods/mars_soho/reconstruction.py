"""Deterministic spherical moment reconstruction for dynamic SOHO maps."""

from __future__ import annotations

import math

import torch

from .statistics import MomentSnapshot


MODEL_MODES = {
    "shared_gaussian",
    "heterogeneous_spherical",
    "support_aware",
    "shuffled_support",
}


def _seed(base_seed: int, class_id: int, stream: int) -> int:
    modulus = 2**63 - 1
    return int((base_seed + 1_000_003 * (class_id + 1) + 97_409 * stream) % modulus)


def _antithetic_normal(
    rows: int,
    columns: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if rows <= 0 or columns <= 0:
        return torch.empty((rows, columns), device=device, dtype=dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    half = (rows + 1) // 2
    positive = torch.randn(
        (half, columns), generator=generator, device=device, dtype=dtype
    )
    return torch.cat((positive, -positive), dim=0)[:rows]


class SphericalReconstructor:
    """Reconstruct deterministic class directions from aggregate moments."""

    def __init__(
        self,
        snapshot: MomentSnapshot,
        *,
        covariance_rank: int,
        shrinkage: float,
        seed: int,
        variance_floor: float = 1e-8,
    ) -> None:
        if snapshot.num_classes <= 0:
            raise ValueError("reconstruction requires at least one class")
        if not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must be in [0, 1]")
        if covariance_rank <= 0:
            raise ValueError("covariance_rank must be positive")
        self.snapshot = snapshot
        self.rank = min(int(covariance_rank), snapshot.feature_dim)
        self.shrinkage = float(shrinkage)
        self.seed = int(seed)
        self.variance_floor = float(variance_floor)
        pooled = (snapshot.pooled_covariance() + snapshot.pooled_covariance().T) * 0.5
        pooled_variance = pooled.diagonal().clamp_min(self.variance_floor)
        pooled_std = pooled_variance.sqrt()
        correlation = pooled / torch.outer(pooled_std, pooled_std)
        correlation = (correlation + correlation.T) * 0.5
        correlation.fill_diagonal_(1)
        eye = torch.eye(
            snapshot.feature_dim, device=pooled.device, dtype=pooled.dtype
        )
        correlation = (1 - self.shrinkage) * correlation + self.shrinkage * eye
        eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
        order = torch.argsort(eigenvalues, descending=True)[: self.rank]
        retained = eigenvalues[order].clamp_min(0)
        self.correlation_factor = eigenvectors[:, order] * retained.sqrt().unsqueeze(0)
        represented_diagonal = self.correlation_factor.square().sum(dim=1)
        self.diagonal_residual = (1 - represented_diagonal).clamp_min(0).sqrt()
        self.pooled_std = pooled_std

    def generate(self, class_id: int, count: int, *, heterogeneous: bool) -> torch.Tensor:
        if count <= 0:
            raise ValueError("pseudo count must be positive")
        column = self.snapshot.class_column(class_id)
        mean = self.snapshot.means[:, column]
        if heterogeneous:
            standard_deviation = self.snapshot.variances[:, column].clamp_min(
                self.variance_floor
            ).sqrt()
        else:
            standard_deviation = self.pooled_std
        low = _antithetic_normal(
            count,
            self.rank,
            seed=_seed(self.seed, class_id, 0),
            device=mean.device,
            dtype=mean.dtype,
        )
        diagonal = _antithetic_normal(
            count,
            self.snapshot.feature_dim,
            seed=_seed(self.seed, class_id, 1),
            device=mean.device,
            dtype=mean.dtype,
        )
        correlated = low @ self.correlation_factor.T
        correlated.add_(diagonal * self.diagonal_residual.unsqueeze(0))
        values = mean.unsqueeze(0) + correlated * standard_deviation.unsqueeze(0)
        return torch.nn.functional.normalize(values, p=2, dim=1)


def allocate_pseudo_budget(
    class_ids: list[int],
    class_counts: torch.Tensor,
    risks: torch.Tensor,
    *,
    total_budget: int,
    minimum_per_class: int,
    risk_floor: float,
) -> dict[int, int]:
    """Allocate a fixed budget using n_c sqrt(boundary-risk + floor)."""
    count = len(class_ids)
    if class_counts.shape != (count,) or risks.shape != (count,):
        raise ValueError("class_counts and risks must align with class_ids")
    if minimum_per_class <= 0 or total_budget < count * minimum_per_class:
        raise ValueError("total budget cannot satisfy minimum_per_class")
    if risk_floor <= 0:
        raise ValueError("risk_floor must be positive")
    remaining = total_budget - count * minimum_per_class
    scores = class_counts.double() * (risks.double() + risk_floor).sqrt()
    if float(scores.sum().item()) == 0:
        scores = torch.ones_like(scores)
    quotas = scores / scores.sum() * remaining
    extras = torch.floor(quotas).long()
    leftover = remaining - int(extras.sum().item())
    if leftover:
        fractions = quotas - extras
        # Stable tie-break by canonical class order.
        order = sorted(range(count), key=lambda index: (-float(fractions[index]), class_ids[index]))
        for index in order[:leftover]:
            extras[index] += 1
    return {
        class_id: minimum_per_class + int(extras[index].item())
        for index, class_id in enumerate(class_ids)
    }


def shuffled_risks(risks: torch.Tensor, *, seed: int) -> torch.Tensor:
    if risks.ndim != 1:
        raise ValueError("risks must be one-dimensional")
    if risks.numel() <= 1:
        return risks.clone()
    generator = torch.Generator(device=risks.device)
    generator.manual_seed(int(seed))
    return risks[torch.randperm(risks.numel(), generator=generator, device=risks.device)]
