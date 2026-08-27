"""Exemplar-free class sketches and reconstruction on the unit sphere."""

from __future__ import annotations

import math

import torch


def _class_seed(base_seed: int, class_id: int, stream: int) -> int:
    modulus = 2**63 - 1
    return int((base_seed + 1_000_003 * (class_id + 1) + 97_409 * stream) % modulus)


def _canonicalize_signs(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    result = matrix.clone()
    pivots = result.abs().argmax(dim=0)
    signs = torch.sign(
        result[pivots, torch.arange(result.shape[1], device=result.device)]
    )
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return result * signs.unsqueeze(0)


def sphere_log_map(points: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Riemannian logarithm on the unit sphere at ``base``."""
    if points.ndim != 2 or base.shape != (points.shape[1],):
        raise ValueError("points/base must have shapes (N,D) and (D,)")
    values = torch.nn.functional.normalize(points, p=2, dim=1)
    origin = torch.nn.functional.normalize(base, p=2, dim=0)
    cosine = (values @ origin).clamp(-1 + 1e-7, 1)
    angle = torch.acos(cosine)
    tangent = values - cosine.unsqueeze(1) * origin.unsqueeze(0)
    tangent_norm = tangent.norm(dim=1)
    scale = torch.where(
        tangent_norm > 1e-10,
        angle / tangent_norm.clamp_min(1e-10),
        torch.zeros_like(angle),
    )
    return tangent * scale.unsqueeze(1)


def sphere_exp_map(tangent: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Riemannian exponential on the unit sphere at ``base``."""
    if tangent.ndim != 2 or base.shape != (tangent.shape[1],):
        raise ValueError("tangent/base must have shapes (N,D) and (D,)")
    origin = torch.nn.functional.normalize(base, p=2, dim=0)
    projected = tangent - (tangent @ origin).unsqueeze(1) * origin.unsqueeze(0)
    norm = projected.norm(dim=1)
    sinc = torch.where(
        norm > 1e-10,
        torch.sin(norm) / norm.clamp_min(1e-10),
        torch.ones_like(norm),
    )
    points = (
        torch.cos(norm).unsqueeze(1) * origin.unsqueeze(0)
        + sinc.unsqueeze(1) * projected
    )
    return torch.nn.functional.normalize(points, p=2, dim=1)


def _antithetic_normal(
    rows: int,
    columns: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    half = (rows + 1) // 2
    positive = torch.randn(
        (half, columns), generator=generator, device=device, dtype=dtype
    )
    return torch.cat((positive, -positive), dim=0)[:rows]


def _randomized_tangent_sketch(
    centered: torch.Tensor,
    rank: int,
    *,
    seed: int,
    oversampling: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return low-rank covariance factors plus an exact diagonal residual."""
    if centered.ndim != 2 or centered.shape[0] < 2:
        raise ValueError("centered samples must have shape (N,D), N >= 2")
    sample_count, dimension = centered.shape
    active_rank = min(int(rank), sample_count - 1, dimension)
    if active_rank <= 0:
        raise ValueError("rank must be positive")
    sketch_rank = min(active_rank + int(oversampling), sample_count - 1, dimension)
    generator = torch.Generator(device=centered.device)
    generator.manual_seed(int(seed))
    omega = torch.randn(
        (dimension, sketch_rank),
        generator=generator,
        device=centered.device,
        dtype=centered.dtype,
    )
    sample = centered.T @ (centered @ omega)
    basis, _ = torch.linalg.qr(sample, mode="reduced")
    # One deterministic subspace iteration improves the spectral approximation.
    sample = centered.T @ (centered @ basis)
    basis, _ = torch.linalg.qr(sample, mode="reduced")
    small = centered @ basis / math.sqrt(sample_count - 1)
    _, singular_values, right = torch.linalg.svd(small, full_matrices=False)
    vectors = _canonicalize_signs(
        basis @ right[:active_rank].T
    )
    eigenvalues = singular_values[:active_rank].square().clamp_min(0)
    diagonal = centered.square().sum(dim=0) / (sample_count - 1)
    represented = (vectors.square() * eigenvalues.unsqueeze(0)).sum(dim=1)
    residual = (diagonal - represented).clamp_min(0)
    return vectors, eigenvalues, residual


class TangentClassSketch:
    """Fixed-rank per-class tangent covariance with no sample-level state."""

    is_exemplar_free = True

    def __init__(
        self,
        *,
        feature_dim: int,
        rank: int,
        calibrated: bool,
        seed: int = 2025,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if feature_dim <= 0 or rank <= 0:
            raise ValueError("feature_dim and rank must be positive")
        self.feature_dim = int(feature_dim)
        self.rank = min(int(rank), self.feature_dim)
        self.calibrated = bool(calibrated)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype
        self.class_ids: list[int] = []
        self.counts = torch.zeros(0, device=self.device, dtype=self.dtype)
        self.mean_directions = torch.zeros(
            (0, self.feature_dim), device=self.device, dtype=self.dtype
        )
        self.resultant_lengths = torch.zeros(0, device=self.device, dtype=self.dtype)
        self.tangent_means = torch.zeros_like(self.mean_directions)
        self.bases = torch.zeros(
            (0, self.feature_dim, self.rank), device=self.device, dtype=self.dtype
        )
        self.eigenvalues = torch.zeros(
            (0, self.rank), device=self.device, dtype=self.dtype
        )
        self.diagonal_residuals = torch.zeros_like(self.mean_directions)
        self.calibration_scales = torch.zeros(0, device=self.device, dtype=self.dtype)

    def _tangent_draws(
        self, column: int, count: int, *, stream_offset: int
    ) -> torch.Tensor:
        class_id = self.class_ids[column]
        low = _antithetic_normal(
            count,
            self.rank,
            seed=_class_seed(self.seed, class_id, 2 * stream_offset),
            device=self.device,
            dtype=self.dtype,
        )
        diagonal = _antithetic_normal(
            count,
            self.feature_dim,
            seed=_class_seed(self.seed, class_id, 2 * stream_offset + 1),
            device=self.device,
            dtype=self.dtype,
        )
        tangent = self.tangent_means[column].unsqueeze(0)
        tangent = tangent + (
            low * self.eigenvalues[column].sqrt().unsqueeze(0)
        ) @ self.bases[column].T
        tangent = tangent + diagonal * self.diagonal_residuals[column].sqrt()
        origin = self.mean_directions[column]
        return tangent - (tangent @ origin).unsqueeze(1) * origin.unsqueeze(0)

    def _calibrate(self, column: int) -> float:
        if not self.calibrated:
            return 1.0
        tangent = self._tangent_draws(column, 256, stream_offset=97)
        origin = self.mean_directions[column]
        target = float(self.resultant_lengths[column].clamp(0, 1).item())
        maximum_norm = float(tangent.norm(dim=1).max().item())
        if maximum_norm <= 1e-12 or target >= 1 - 1e-7:
            return 0.0
        lower, upper = 0.0, min(4.0, (math.pi - 1e-3) / maximum_norm)
        for _ in range(32):
            middle = (lower + upper) * 0.5
            resultant = float(
                sphere_exp_map(tangent * middle, origin).mean(dim=0).norm().item()
            )
            if resultant > target:
                lower = middle
            else:
                upper = middle
        return (lower + upper) * 0.5

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        progress: bool = False,
    ) -> None:
        values = torch.nn.functional.normalize(
            features.to(self.device, self.dtype), p=2, dim=1
        )
        targets = labels.to(self.device, torch.long)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (N,{self.feature_dim})")
        if targets.ndim != 1 or targets.shape[0] != values.shape[0]:
            raise ValueError("labels must align with features")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        self.class_ids = sorted(map(int, torch.unique(targets).tolist()))
        counts, directions, resultants = [], [], []
        tangent_means, bases, eigenvalues, residuals = [], [], [], []
        for class_index, class_id in enumerate(self.class_ids, start=1):
            class_values = values[targets == class_id]
            if class_values.shape[0] <= self.rank:
                raise ValueError("each class needs more samples than sketch rank")
            euclidean_mean = class_values.mean(dim=0)
            resultant = euclidean_mean.norm().clamp(0, 1)
            if float(resultant.item()) <= torch.finfo(self.dtype).eps:
                raise ValueError("class mean direction is numerically undefined")
            direction = torch.nn.functional.normalize(euclidean_mean, p=2, dim=0)
            tangent = sphere_log_map(class_values, direction)
            tangent_mean = tangent.mean(dim=0)
            centered = tangent - tangent_mean.unsqueeze(0)
            basis, spectrum, residual = _randomized_tangent_sketch(
                centered,
                self.rank,
                seed=_class_seed(self.seed, class_id, 41),
            )
            counts.append(float(class_values.shape[0]))
            directions.append(direction)
            resultants.append(resultant)
            tangent_means.append(tangent_mean)
            bases.append(basis)
            eigenvalues.append(spectrum)
            residuals.append(residual)
            if progress and (
                class_index == 1
                or class_index % 10 == 0
                or class_index == len(self.class_ids)
            ):
                print(
                    f"SKETCH class={class_index}/{len(self.class_ids)} "
                    f"rank={self.rank} calibrated={self.calibrated}",
                    flush=True,
                )
        self.counts = torch.tensor(counts, device=self.device, dtype=self.dtype)
        self.mean_directions = torch.stack(directions)
        self.resultant_lengths = torch.stack(resultants)
        self.tangent_means = torch.stack(tangent_means)
        self.bases = torch.stack(bases)
        self.eigenvalues = torch.stack(eigenvalues)
        self.diagonal_residuals = torch.stack(residuals)
        self.calibration_scales = torch.tensor(
            [self._calibrate(index) for index in range(len(self.class_ids))],
            device=self.device,
            dtype=self.dtype,
        )
        self.assert_exemplar_free_state()

    def generate(
        self, class_id: int, count: int, *, stream_offset: int = 0
    ) -> torch.Tensor:
        if count <= 0:
            raise ValueError("count must be positive")
        if class_id not in self.class_ids:
            raise KeyError(class_id)
        column = self.class_ids.index(class_id)
        tangent = self._tangent_draws(column, count, stream_offset=stream_offset)
        tangent = tangent * self.calibration_scales[column]
        return sphere_exp_map(tangent, self.mean_directions[column])

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "class_counts": self.counts,
            "mean_directions": self.mean_directions,
            "resultant_lengths": self.resultant_lengths,
            "tangent_means": self.tangent_means,
            "tangent_bases": self.bases,
            "tangent_eigenvalues": self.eigenvalues,
            "tangent_diagonal_residuals": self.diagonal_residuals,
            "calibration_scales": self.calibration_scales,
        }

    def persistent_state_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.persistent_tensors().values()
        )

    def assert_exemplar_free_state(self) -> None:
        for name, tensor in self.persistent_tensors().items():
            if tensor.ndim and tensor.shape[0] not in {
                len(self.class_ids), self.feature_dim
            }:
                raise AssertionError(f"unexpected leading state dimension: {name}")
            if not bool(torch.isfinite(tensor).all()):
                raise AssertionError(f"non-finite sketch tensor: {name}")
