"""Sample-free class memory and transient WTA-aware residual transport."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from methods.mars_soho.tangent import (
    TangentClassSketch,
    _antithetic_normal,
    _class_seed,
    sphere_exp_map,
    sphere_log_map,
)


def topk_gap(expanded: torch.Tensor, k: int) -> torch.Tensor:
    """Return the order-statistic gap between winners k and k+1."""
    if expanded.ndim != 2 or not 0 < k < expanded.shape[1]:
        raise ValueError("expanded must have shape (N,M) with 0 < k < M")
    winners = torch.topk(expanded, k + 1, dim=1, largest=True).values
    return winners[:, k - 1] - winners[:, k]


def certified_topk_support_stable(
    expanded: torch.Tensor, perturbation: torch.Tensor, k: int
) -> torch.Tensor:
    """Sufficient support-stability certificate for an additive perturbation.

    If the k/(k+1) gap exceeds twice the infinity norm of the preactivation
    perturbation, no loser can overtake a winner.
    """
    if expanded.shape != perturbation.shape:
        raise ValueError("expanded and perturbation must have identical shapes")
    return topk_gap(expanded, k) > 2 * perturbation.abs().amax(dim=1)


def whiten_color_tangent_residuals(
    residuals: torch.Tensor,
    *,
    source_basis: torch.Tensor,
    source_eigenvalues: torch.Tensor,
    source_diagonal_residual: torch.Tensor,
    target_basis: torch.Tensor,
    target_eigenvalues: torch.Tensor,
    target_diagonal_residual: torch.Tensor,
    target_origin: torch.Tensor,
    variance_floor: float = 1e-8,
    standardized_clip: float = 6.0,
) -> torch.Tensor:
    """Whiten source residual coordinates and color them with target moments.

    The low-rank part is exact when both covariance models span the data.  The
    diagonal remainder is a bounded coordinate-wise correction for discarded
    spectral energy.  Returned vectors are tangent at ``target_origin``.
    """
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape (N,D)")
    dimension = residuals.shape[1]
    if (
        source_basis.shape[0] != dimension
        or target_basis.shape[0] != dimension
        or source_diagonal_residual.shape != (dimension,)
        or target_diagonal_residual.shape != (dimension,)
        or target_origin.shape != (dimension,)
    ):
        raise ValueError("transport tensors have incompatible dimensions")
    source_scale = source_eigenvalues.clamp_min(variance_floor).sqrt()
    target_scale = target_eigenvalues.clamp_min(0).sqrt()
    source_coordinates = residuals @ source_basis
    standardized_low = source_coordinates / source_scale.unsqueeze(0)
    colored_low = standardized_low @ (target_basis * target_scale.unsqueeze(0)).T

    source_low = source_coordinates @ source_basis.T
    source_remainder = residuals - source_low
    source_diagonal_scale = source_diagonal_residual.clamp_min(variance_floor).sqrt()
    standardized_diagonal = source_remainder / source_diagonal_scale.unsqueeze(0)
    standardized_diagonal = standardized_diagonal.clamp(
        -standardized_clip, standardized_clip
    )
    colored_diagonal = standardized_diagonal * target_diagonal_residual.clamp_min(0).sqrt()
    transported = colored_low + colored_diagonal
    origin = torch.nn.functional.normalize(target_origin, p=2, dim=0)
    return transported - (transported @ origin).unsqueeze(1) * origin.unsqueeze(0)


@dataclass(frozen=True)
class TransportBatch:
    features: dict[int, torch.Tensor]
    diagnostics: dict[int, dict]


class BoundaryTransportMemory:
    """Per-class tangent sketches; no observation-level tensor is retained."""

    is_exemplar_free = True

    def __init__(
        self,
        *,
        feature_dim: int,
        rank: int,
        seed: int = 2025,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if feature_dim <= 0 or rank <= 0:
            raise ValueError("feature_dim and rank must be positive")
        self.feature_dim = int(feature_dim)
        self.rank = min(int(rank), self.feature_dim)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype
        self.class_ids: list[int] = []
        self.counts = torch.zeros(0, device=self.device, dtype=self.dtype)
        self.mean_directions = torch.zeros(
            (0, self.feature_dim), device=self.device, dtype=self.dtype
        )
        self.tangent_means = torch.zeros_like(self.mean_directions)
        self.bases = torch.zeros(
            (0, self.feature_dim, self.rank), device=self.device, dtype=self.dtype
        )
        self.eigenvalues = torch.zeros(
            (0, self.rank), device=self.device, dtype=self.dtype
        )
        self.diagonal_residuals = torch.zeros_like(self.mean_directions)

    def _fit_transient(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> tuple[TangentClassSketch, torch.Tensor, torch.Tensor]:
        values = torch.nn.functional.normalize(
            features.to(self.device, self.dtype), p=2, dim=1
        )
        targets = labels.to(self.device, torch.long)
        sketch = TangentClassSketch(
            feature_dim=self.feature_dim,
            rank=self.rank,
            calibrated=False,
            seed=self.seed,
            device=self.device,
            dtype=self.dtype,
        )
        sketch.fit(values, targets)
        return sketch, values, targets

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Append statistics for newly arriving, class-disjoint observations."""
        sketch, _, _ = self._fit_transient(features, labels)
        overlap = sorted(set(self.class_ids) & set(sketch.class_ids))
        if overlap:
            raise ValueError(f"WBT expects class-disjoint tasks; repeated classes: {overlap}")
        combined_ids = self.class_ids + sketch.class_ids
        order = sorted(range(len(combined_ids)), key=combined_ids.__getitem__)
        self.class_ids = [combined_ids[index] for index in order]
        self.counts = torch.cat((self.counts, sketch.counts))[order]
        self.mean_directions = torch.cat(
            (self.mean_directions, sketch.mean_directions), dim=0
        )[order]
        self.tangent_means = torch.cat(
            (self.tangent_means, sketch.tangent_means), dim=0
        )[order]
        self.bases = torch.cat((self.bases, sketch.bases), dim=0)[order]
        self.eigenvalues = torch.cat(
            (self.eigenvalues, sketch.eigenvalues), dim=0
        )[order]
        self.diagonal_residuals = torch.cat(
            (self.diagonal_residuals, sketch.diagonal_residuals), dim=0
        )[order]
        self.assert_exemplar_free_state()

    def _column(self, class_id: int) -> int:
        try:
            return self.class_ids.index(int(class_id))
        except ValueError as error:
            raise KeyError(class_id) from error

    def generate_tangent_gaussian(
        self, class_id: int, count: int, *, stream_offset: int = 0
    ) -> torch.Tensor:
        column = self._column(class_id)
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
        tangent = tangent - (tangent @ origin).unsqueeze(1) * origin.unsqueeze(0)
        return sphere_exp_map(tangent, origin)

    def _source_assignments(
        self,
        *,
        target_class_ids: list[int],
        current_sketch: TangentClassSketch,
        encoder,
        shuffled: bool,
        stream_offset: int,
    ) -> dict[int, int]:
        target_directions = torch.stack([
            self.mean_directions[self._column(class_id)]
            for class_id in target_class_ids
        ])
        target_codes = encoder.encode(target_directions)
        current_codes = encoder.encode(current_sketch.mean_directions)
        distances = torch.cdist(target_codes, current_codes).square()
        nearest = distances.argmin(dim=1)
        if shuffled:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed + 104_729 * (stream_offset + 1))
            # Preserve the nearest-enemy marginal counts while destroying the
            # target/enemy correspondence.  This is a true shuffled control,
            # not a different random source distribution.
            permutation = torch.randperm(
                len(target_class_ids), generator=generator, device=self.device
            )
            nearest = nearest[permutation]
        return {
            target: current_sketch.class_ids[int(nearest[index].item())]
            for index, target in enumerate(target_class_ids)
        }

    def _transport_one(
        self,
        *,
        target_class_id: int,
        source_class_id: int,
        current_sketch: TangentClassSketch,
        current_values: torch.Tensor,
        current_labels: torch.Tensor,
        count: int,
        covariance_coloring: bool,
        encoder,
        boundary_fraction: float,
        boundary_strength: float,
        stream_offset: int,
    ) -> tuple[torch.Tensor, dict]:
        target_column = self._column(target_class_id)
        source_column = current_sketch.class_ids.index(source_class_id)
        source_values = current_values[current_labels == source_class_id]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(
            _class_seed(self.seed + stream_offset, target_class_id, source_class_id + 71)
        )
        order = torch.randperm(source_values.shape[0], generator=generator, device=self.device)
        indices = order.repeat((count + len(order) - 1) // len(order))[:count]
        selected = source_values[indices]
        source_origin = current_sketch.mean_directions[source_column]
        target_origin = self.mean_directions[target_column]
        source_tangent = sphere_log_map(selected, source_origin)
        residuals = source_tangent - current_sketch.tangent_means[source_column]
        if covariance_coloring:
            transported = whiten_color_tangent_residuals(
                residuals,
                source_basis=current_sketch.bases[source_column],
                source_eigenvalues=current_sketch.eigenvalues[source_column],
                source_diagonal_residual=current_sketch.diagonal_residuals[source_column],
                target_basis=self.bases[target_column],
                target_eigenvalues=self.eigenvalues[target_column],
                target_diagonal_residual=self.diagonal_residuals[target_column],
                target_origin=target_origin,
            )
        else:
            transported = residuals - (residuals @ target_origin).unsqueeze(1) * target_origin
        transported = transported + self.tangent_means[target_column].unsqueeze(0)
        base_features = sphere_exp_map(transported, target_origin)
        base_gap = topk_gap(encoder.expanded(base_features), encoder.k)
        if boundary_fraction <= 0 or boundary_strength <= 0:
            return base_features, {
                "source_class_id": source_class_id,
                "boundary_rows": 0,
                "mean_topk_gap_before": float(base_gap.mean().item()),
                "mean_topk_gap_after": float(base_gap.mean().item()),
                "old_dominance_fraction": 1.0,
            }

        boundary_count = min(count, max(1, round(count * boundary_fraction)))
        chosen_rows = torch.randperm(count, generator=generator, device=self.device)[
            :boundary_count
        ]
        enemy_origin = current_sketch.mean_directions[source_column]
        boundary_direction = sphere_log_map(enemy_origin.unsqueeze(0), target_origin)[0]
        alpha_grid = torch.linspace(
            0,
            float(boundary_strength),
            6,
            device=self.device,
            dtype=self.dtype,
        )
        candidate_blocks = []
        for alpha in alpha_grid:
            candidate_blocks.append(
                sphere_exp_map(
                    transported[chosen_rows] + alpha * boundary_direction.unsqueeze(0),
                    target_origin,
                )
            )
        candidates = torch.stack(candidate_blocks, dim=1)
        flat = candidates.reshape(-1, self.feature_dim)
        flat_codes = encoder.encode(flat)
        target_code = encoder.encode(target_origin.unsqueeze(0))
        enemy_code = encoder.encode(enemy_origin.unsqueeze(0))
        old_distance = (flat_codes - target_code).square().sum(dim=1).reshape(
            boundary_count, -1
        )
        enemy_distance = (flat_codes - enemy_code).square().sum(dim=1).reshape(
            boundary_count, -1
        )
        dominance = old_distance <= enemy_distance
        gaps = topk_gap(encoder.expanded(flat), encoder.k).reshape(boundary_count, -1)
        penalized = torch.where(dominance, gaps, torch.full_like(gaps, torch.inf))
        selected_alpha = penalized.argmin(dim=1)
        row_ids = torch.arange(boundary_count, device=self.device)
        moved = candidates[row_ids, selected_alpha]
        output = base_features.clone()
        output[chosen_rows] = moved
        final_expanded = encoder.expanded(output)
        final_gap = topk_gap(final_expanded, encoder.k)
        final_codes = encoder.encode(output[chosen_rows])
        final_old = (final_codes - target_code).square().sum(dim=1)
        final_enemy = (final_codes - enemy_code).square().sum(dim=1)
        return output, {
            "source_class_id": source_class_id,
            "boundary_rows": boundary_count,
            "mean_topk_gap_before": float(base_gap.mean().item()),
            "mean_topk_gap_after": float(final_gap.mean().item()),
            "old_dominance_fraction": float((final_old <= final_enemy).float().mean().item()),
        }

    def transport(
        self,
        *,
        current_features: torch.Tensor,
        current_labels: torch.Tensor,
        target_class_ids: list[int],
        count: int,
        encoder,
        mode: str,
        boundary_fraction: float = 0.0,
        boundary_strength: float = 0.0,
        stream_offset: int = 0,
    ) -> TransportBatch:
        if mode not in {
            "mean_shift_empirical",
            "covariance_transport",
            "wta_boundary_transport",
            "shuffled_enemy_boundary_transport",
        }:
            raise ValueError(f"unsupported transport mode: {mode}")
        if count <= 0 or not 0 <= boundary_fraction <= 1 or boundary_strength < 0:
            raise ValueError("invalid transport count or boundary parameters")
        current_sketch, values, labels = self._fit_transient(
            current_features, current_labels
        )
        shuffled = mode == "shuffled_enemy_boundary_transport"
        assignments = self._source_assignments(
            target_class_ids=target_class_ids,
            current_sketch=current_sketch,
            encoder=encoder,
            shuffled=shuffled,
            stream_offset=stream_offset,
        )
        covariance_coloring = mode != "mean_shift_empirical"
        use_boundary = mode in {
            "wta_boundary_transport", "shuffled_enemy_boundary_transport"
        }
        features, diagnostics = {}, {}
        for class_id in target_class_ids:
            pseudo, audit = self._transport_one(
                target_class_id=class_id,
                source_class_id=assignments[class_id],
                current_sketch=current_sketch,
                current_values=values,
                current_labels=labels,
                count=count,
                covariance_coloring=covariance_coloring,
                encoder=encoder,
                boundary_fraction=boundary_fraction if use_boundary else 0.0,
                boundary_strength=boundary_strength if use_boundary else 0.0,
                stream_offset=stream_offset,
            )
            features[class_id] = pseudo
            diagnostics[class_id] = audit
        return TransportBatch(features=features, diagnostics=diagnostics)

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "transport_class_counts": self.counts,
            "transport_mean_directions": self.mean_directions,
            "transport_tangent_means": self.tangent_means,
            "transport_tangent_bases": self.bases,
            "transport_tangent_eigenvalues": self.eigenvalues,
            "transport_diagonal_residuals": self.diagonal_residuals,
        }

    def persistent_state_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in self.persistent_tensors().values()
        )

    def assert_exemplar_free_state(self) -> None:
        class_count = len(self.class_ids)
        expected = {
            "transport_class_counts": (class_count,),
            "transport_mean_directions": (class_count, self.feature_dim),
            "transport_tangent_means": (class_count, self.feature_dim),
            "transport_tangent_bases": (class_count, self.feature_dim, self.rank),
            "transport_tangent_eigenvalues": (class_count, self.rank),
            "transport_diagonal_residuals": (class_count, self.feature_dim),
        }
        for name, tensor in self.persistent_tensors().items():
            if tensor.shape != expected[name]:
                raise AssertionError(f"unexpected WBT state shape: {name}")
            if not bool(torch.isfinite(tensor).all()):
                raise AssertionError(f"non-finite WBT state: {name}")
