"""Streaming WBT-SOHO learner used by the train-only Phase-1D gate."""

from __future__ import annotations

import inspect

import torch

from methods.mars_soho.learner import DynamicSOHOMap, _solve_ridge
from methods.mars_soho.statistics import SphericalClassMoments

from .transport import BoundaryTransportMemory


WBTMODES = {
    "tangent_gaussian",
    "mean_shift_empirical",
    "covariance_transport",
    "wta_boundary_transport",
    "shuffled_enemy_boundary_transport",
}


class WBTSOHOLearner:
    """Dynamic SOHO with transient current-task residual transport."""

    is_exemplar_free = True

    def __init__(
        self,
        *,
        feature_dim: int,
        expand_dim: int,
        density: float,
        olda_dim: int,
        use_etf: bool,
        coding_level: float,
        ridge_lambda: float,
        tangent_rank: int,
        pseudo_per_class: int,
        mode: str,
        boundary_fraction: float = 0.0,
        boundary_strength: float = 0.0,
        seed: int = 2025,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if mode not in WBTMODES:
            raise ValueError(f"mode must be one of {sorted(WBTMODES)}")
        if ridge_lambda <= 0 or tangent_rank <= 0 or pseudo_per_class <= 0:
            raise ValueError("ridge, rank and pseudo count must be positive")
        self.feature_dim = int(feature_dim)
        self.expand_dim = int(expand_dim)
        self.olda_dim = min(int(olda_dim), self.feature_dim)
        self.density = float(density)
        self.use_etf = bool(use_etf)
        self.coding_level = float(coding_level)
        self.ridge_lambda = float(ridge_lambda)
        self.tangent_rank = int(tangent_rank)
        self.pseudo_per_class = int(pseudo_per_class)
        self.mode = mode
        self.boundary_fraction = float(boundary_fraction)
        self.boundary_strength = float(boundary_strength)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype
        self.moments = SphericalClassMoments(
            self.feature_dim, device=self.device, dtype=self.dtype
        )
        self.transport_memory = BoundaryTransportMemory(
            feature_dim=self.feature_dim,
            rank=self.tangent_rank,
            seed=self.seed,
            device=self.device,
            dtype=self.dtype,
        )
        self.encoder = DynamicSOHOMap(
            feature_dim=self.feature_dim,
            expand_dim=self.expand_dim,
            density=self.density,
            olda_dim=self.olda_dim,
            coding_level=self.coding_level,
            use_etf=self.use_etf,
            seed=self.seed,
            device=self.device,
            dtype=self.dtype,
        )
        self.G = torch.zeros(
            (self.expand_dim, self.expand_dim), device=self.device, dtype=self.dtype
        )
        self.Q = torch.zeros((self.expand_dim, 0), device=self.device, dtype=self.dtype)
        self.weights: torch.Tensor | None = None
        self.diagnostics: dict = {"phase": "1D", "mode": self.mode}

    @property
    def class_ids(self) -> list[int]:
        return list(self.moments.class_ids)

    def _targets(self, labels: torch.Tensor) -> torch.Tensor:
        columns = {class_id: index for index, class_id in enumerate(self.class_ids)}
        indices = torch.tensor(
            [columns[int(value)] for value in labels.detach().cpu().tolist()],
            device=self.device,
        )
        return torch.nn.functional.one_hot(
            indices, num_classes=len(self.class_ids)
        ).to(self.dtype)

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        values = features.to(self.device, self.dtype)
        targets = labels.to(self.device, torch.long)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (N,{self.feature_dim})")
        if targets.shape != (values.shape[0],):
            raise ValueError("labels must align with features")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        arriving_classes = sorted(map(int, torch.unique(targets).tolist()))
        overlap = sorted(set(arriving_classes) & set(self.transport_memory.class_ids))
        if overlap:
            raise ValueError(f"WBT expects class-disjoint tasks; repeated classes: {overlap}")
        old_class_ids = list(self.transport_memory.class_ids)
        old_counts = self.transport_memory.counts.clone()
        self.moments.update(values, targets)
        self.encoder.update_rotation(self.moments.snapshot())
        self.G.zero_()
        self.Q = torch.zeros(
            (self.expand_dim, len(self.class_ids)),
            device=self.device,
            dtype=self.dtype,
        )
        current_codes = self.encoder.encode(values)
        self.G.add_(current_codes.T @ current_codes)
        self.Q.add_(current_codes.T @ self._targets(targets))
        transport_diagnostics: dict[int, dict] = {}
        if old_class_ids:
            if self.mode == "tangent_gaussian":
                pseudo = {
                    class_id: self.transport_memory.generate_tangent_gaussian(
                        class_id,
                        self.pseudo_per_class,
                        stream_offset=self.moments.total_count,
                    )
                    for class_id in old_class_ids
                }
            else:
                transported = self.transport_memory.transport(
                    current_features=values,
                    current_labels=targets,
                    target_class_ids=old_class_ids,
                    count=self.pseudo_per_class,
                    encoder=self.encoder,
                    mode=self.mode,
                    boundary_fraction=self.boundary_fraction,
                    boundary_strength=self.boundary_strength,
                    stream_offset=self.moments.total_count,
                )
                pseudo = transported.features
                transport_diagnostics = transported.diagnostics
            columns = {class_id: index for index, class_id in enumerate(self.class_ids)}
            blocks, row_weights = [], []
            for old_index, class_id in enumerate(old_class_ids):
                codes = self.encoder.encode(pseudo[class_id])
                weight = old_counts[old_index] / self.pseudo_per_class
                blocks.append(codes)
                row_weights.append(torch.full(
                    (self.pseudo_per_class,),
                    weight,
                    device=self.device,
                    dtype=self.dtype,
                ))
                self.Q[:, columns[class_id]].add_(weight * codes.sum(dim=0))
            all_codes = torch.cat(blocks)
            all_weights = torch.cat(row_weights)
            weighted = all_codes * all_weights.sqrt().unsqueeze(1)
            self.G.add_(weighted.T @ weighted)
        self.transport_memory.update(values, targets)
        self.weights, residual = _solve_ridge(self.G, self.Q, self.ridge_lambda)
        gaps_before = [
            item["mean_topk_gap_before"] for item in transport_diagnostics.values()
        ]
        gaps_after = [
            item["mean_topk_gap_after"] for item in transport_diagnostics.values()
        ]
        dominance = [
            item["old_dominance_fraction"] for item in transport_diagnostics.values()
        ]
        self.diagnostics = {
            "phase": "1D",
            "mode": self.mode,
            "total_count": self.moments.total_count,
            "old_class_count": len(old_class_ids),
            "pseudo_total": len(old_class_ids) * self.pseudo_per_class,
            "solver_relative_residual": residual,
            "mean_topk_gap_before": sum(gaps_before) / len(gaps_before)
            if gaps_before else None,
            "mean_topk_gap_after": sum(gaps_after) / len(gaps_after)
            if gaps_after else None,
            # Use the worst old class, so the outer gate cannot be passed by
            # averaging a boundary-crossing class with easy classes.
            "old_dominance_fraction": min(dominance)
            if dominance else None,
            "transport_sources": {
                str(class_id): item["source_class_id"]
                for class_id, item in transport_diagnostics.items()
            },
        }
        self.assert_exemplar_free_state()

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("update() must be called before prediction")
        return self.encoder.encode(features) @ self.weights

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(features).argmax(dim=1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[index] for index in columns])

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "random_projection": self.encoder.projection,
            "active_rotation": self.encoder.rotation,
            "gram": self.G,
            "cross": self.Q,
            **self.moments.persistent_tensors(),
            **self.transport_memory.persistent_tensors(),
        }
        if self.weights is not None:
            tensors["classifier"] = self.weights
        return tensors

    def persistent_state_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in self.persistent_tensors().values()
        )

    def assert_exemplar_free_state(self) -> None:
        self.transport_memory.assert_exemplar_free_state()
        forbidden = ("history", "sample", "image", "dataset", "loader", "cache")
        offending = [
            name for name in self.persistent_tensors()
            if any(token in name.lower() for token in forbidden)
        ]
        if offending:
            raise AssertionError(f"forbidden persistent tensors: {offending}")
        historical_count = self.moments.total_count
        structural = {
            self.feature_dim,
            self.expand_dim,
            self.olda_dim,
            len(self.class_ids),
        }
        if historical_count > max(structural):
            for name, value in self.persistent_tensors().items():
                if historical_count in value.shape:
                    raise AssertionError(
                        f"{name} has a historical sample-count dimension"
                    )


assert "task_id" not in inspect.signature(WBTSOHOLearner.predict_logits).parameters
