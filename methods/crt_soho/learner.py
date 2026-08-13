"""Exemplar-free CRT-SOHO learner built from exact sufficient statistics."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from models.flyhash import FlyHash

from .geometry import (
    anchor_weights,
    fisher_directions,
    pairwise_between,
    random_directions,
    raw_scatter,
    relative_margin_affinity,
    shuffled_affinity,
)
from .solver import reconstruct_residual_statistics, schur_residual_directions, solve_block_ridge
from .statistics import DualViewStatistics


METHODS = {
    "anchor_only",
    "full_raw_residual",
    "random_residual",
    "fisher_residual",
    "confusion_residual",
    "shuffled_confusion_residual",
    "confusion_no_residualization",
    "schur_residual",
}


class _RestoredFlyAnchor(nn.Module):
    """FlyHash forward semantics with an already materialized projection."""

    forward = FlyHash.forward

    def __init__(self, projection: torch.Tensor, raw_dim: int, anchor_dim: int, synaptic_degree: int):
        super().__init__()
        self.in_dim = raw_dim
        self.expand_dim = anchor_dim
        self.synaptic_degree = synaptic_degree
        self.register_buffer("projection_matrix", projection)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.sparse_csc:
        return sum(part.numel() * part.element_size() for part in (
            tensor.ccol_indices(), tensor.row_indices(), tensor.values()
        ))
    if tensor.layout == torch.sparse_coo:
        tensor = tensor.coalesce()
        return sum(part.numel() * part.element_size() for part in (tensor.indices(), tensor.values()))
    return tensor.numel() * tensor.element_size()


class CRTSOHOLearner:
    """Fixed nonlinear anchor plus a statistic-only residual branch.

    ``predict_logits`` deliberately has no task identifier. Retained tensors
    are bounded by feature dimensions and seen-class count, never N_seen.
    """

    is_exemplar_free = True

    def __init__(
        self,
        method: str,
        raw_dim: int,
        anchor_dim: int,
        synaptic_degree: int,
        coding_level: float,
        anchor_ridge: float,
        residual_ridge: float = 1.0,
        complement_ridge: float = 1.0,
        requested_rank: int = 64,
        confusion_temperature: float = 1.0,
        scatter_epsilon: float = 1e-4,
        seed: int = 1993,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        anchor_projection: torch.Tensor | None = None,
    ):
        if method not in METHODS:
            raise ValueError(f"unknown CRT-SOHO method {method!r}; choices: {sorted(METHODS)}")
        if not 0 < coding_level <= 1:
            raise ValueError("coding_level must be in (0, 1]")
        if int(anchor_dim * coding_level) < 1:
            raise ValueError("coding_level must retain at least one anchor coordinate")
        if not 0 < synaptic_degree <= raw_dim:
            raise ValueError("synaptic_degree must be in [1, raw_dim]")
        if min(anchor_ridge, residual_ridge, complement_ridge, scatter_epsilon) <= 0:
            raise ValueError("all regularization coefficients must be positive")
        if requested_rank <= 0:
            raise ValueError("requested_rank must be positive")
        if confusion_temperature <= 0:
            raise ValueError("confusion_temperature must be positive")

        self.method = method
        self.raw_dim = int(raw_dim)
        self.anchor_dim = int(anchor_dim)
        self.synaptic_degree = int(synaptic_degree)
        self.coding_level = float(coding_level)
        self.anchor_ridge = float(anchor_ridge)
        self.residual_ridge = float(residual_ridge)
        self.complement_ridge = float(complement_ridge)
        self.requested_rank = int(requested_rank)
        self.confusion_temperature = float(confusion_temperature)
        self.scatter_epsilon = float(scatter_epsilon)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype

        if anchor_projection is None:
            fork_devices: list[int] = []
            if self.device.type == "cuda":
                fork_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(self.seed)
                anchor = FlyHash(self.raw_dim, self.anchor_dim, self.synaptic_degree)
            anchor.to_sparse()
        else:
            if anchor_projection.shape != (self.anchor_dim, self.raw_dim):
                raise ValueError("invalid anchor projection shape")
            if anchor_projection.layout != torch.sparse_csc:
                raise ValueError("anchor projection must use sparse CSC layout")
            if not bool(torch.isfinite(anchor_projection.values()).all()):
                raise ValueError("anchor projection contains NaN or Inf")
            anchor = _RestoredFlyAnchor(
                anchor_projection, self.raw_dim, self.anchor_dim, self.synaptic_degree
            )
        self.anchor = anchor.to(self.device)
        self.statistics = DualViewStatistics(self.raw_dim, self.anchor_dim, self.device, self.dtype)

        self.anchor_classifier: torch.Tensor | None = None
        self.directions: torch.Tensor | None = None
        self.complement: torch.Tensor | None = None
        self.residual_classifier: torch.Tensor | None = None
        self.diagnostics: dict = {}

    @property
    def class_ids(self) -> list[int]:
        return self.statistics.class_ids

    def _anchor_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        x = raw_features.to(device=self.device, dtype=self.anchor.projection_matrix.dtype)
        if x.ndim != 2 or x.shape[1] != self.raw_dim:
            raise ValueError(f"features must have shape (B, {self.raw_dim})")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("features contain NaN or Inf")
        return self.anchor(x, self.coding_level).to(dtype=self.dtype)

    def encode_anchor(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Return the fixed nonlinear view for experiment-cache construction."""
        return self._anchor_features(raw_features)

    def _residual_rank(self) -> int:
        if self.method == "full_raw_residual":
            return self.raw_dim
        if self.method == "schur_residual":
            return min(self.requested_rank, self.raw_dim, max(self.statistics.num_classes, 1))
        return min(self.requested_rank, self.raw_dim, max(self.statistics.num_classes - 1, 1))

    def _directions_and_geometry(self, raw_anchor_weights: torch.Tensor) -> tuple[torch.Tensor, dict]:
        rank = self._residual_rank()
        if self.method == "full_raw_residual":
            return torch.eye(self.raw_dim, device=self.device, dtype=self.dtype), {
                "geometry": "identity", "requested_rank": self.requested_rank,
                "effective_rank": rank,
            }
        if self.method == "random_residual":
            return random_directions(self.raw_dim, rank, self.seed, self.dtype, self.device), {
                "geometry": "random", "requested_rank": self.requested_rank,
                "effective_rank": rank,
            }
        if self.method == "schur_residual":
            geometry = schur_residual_directions(
                self.statistics,
                self.complement_ridge,
                self.anchor_ridge,
                self.residual_ridge,
                rank,
            )
            return geometry.directions, {
                "geometry": "schur_targeted",
                "requested_rank": self.requested_rank,
                "effective_rank": geometry.effective_rank,
                "singular_values": geometry.singular_values.detach().clone(),
                "retained_correction_energy": geometry.retained_correction_energy,
            }

        within, standard_between, means = raw_scatter(self.statistics)
        affinity = None
        between = standard_between
        geometry = "standard_fisher"
        if self.method in {
            "confusion_residual", "shuffled_confusion_residual", "confusion_no_residualization",
        }:
            affinity = relative_margin_affinity(
                self.statistics, raw_anchor_weights, self.confusion_temperature
            )
            if self.method == "shuffled_confusion_residual":
                affinity = shuffled_affinity(affinity, self.seed)
            between = pairwise_between(means, self.statistics.counts, affinity)
            geometry = "shuffled_confusion_fisher" if "shuffled" in self.method else "confusion_fisher"
        directions, eigenvalues = fisher_directions(
            within, between, self.statistics.total_count, self.statistics.num_classes,
            rank, self.scatter_epsilon,
        )
        result = {
            "geometry": geometry,
            "requested_rank": self.requested_rank,
            "effective_rank": rank,
            "eigenvalues": eigenvalues.detach().clone(),
        }
        if affinity is not None and affinity.shape[0] > 1:
            rows, columns = torch.triu_indices(
                affinity.shape[0], affinity.shape[1], 1, device=self.device
            )
            edges = affinity[rows, columns]
            edge_mean = edges.mean()
            probabilities = edges / edges.sum().clamp_min(torch.finfo(edges.dtype).tiny)
            entropy = -(probabilities * probabilities.clamp_min(torch.finfo(edges.dtype).tiny).log()).sum()
            maximum_entropy = torch.log(torch.tensor(float(edges.numel()), device=self.device, dtype=self.dtype))
            result.update(
                affinity_min=float(edges.min().item()),
                affinity_median=float(edges.median().item()),
                affinity_max=float(edges.max().item()),
                affinity_edge_cv=float((edges.std(unbiased=False) / edge_mean.clamp_min(torch.finfo(edges.dtype).tiny)).item()),
                affinity_normalized_entropy=float((entropy / maximum_entropy.clamp_min(1)).item()),
            )
        return directions, result

    def _recompute(self) -> None:
        raw_anchor_weights = anchor_weights(self.statistics, self.anchor_ridge)
        if self.method == "anchor_only":
            eye = torch.eye(self.anchor_dim, device=self.device, dtype=self.dtype)
            residual = (self.statistics.G_pp + self.anchor_ridge * eye) @ raw_anchor_weights
            self.anchor_classifier = raw_anchor_weights
            self.directions = self.complement = self.residual_classifier = None
            self.diagnostics = {
                "geometry": "anchor_only",
                "requested_rank": self.requested_rank,
                "effective_rank": 0,
                "solver_residual_max": float((residual - self.statistics.Q_p).abs().max().item()),
            }
            target_scale = max(float(self.statistics.Q_p.abs().max().item()), 1.0)
            self.diagnostics["solver_relative_residual_max"] = (
                self.diagnostics["solver_residual_max"] / target_scale
            )
            return

        directions, diagnostics = self._directions_and_geometry(raw_anchor_weights)
        residualized = self.method != "confusion_no_residualization"
        residual = reconstruct_residual_statistics(
            self.statistics, directions, self.complement_ridge, residualize=residualized
        )
        anchor_classifier, residual_classifier, solver_residual = solve_block_ridge(
            self.statistics, residual, self.anchor_ridge, self.residual_ridge
        )
        self.anchor_classifier = anchor_classifier
        self.directions = directions
        self.complement = residual.C
        self.residual_classifier = residual_classifier
        self.diagnostics = {
            **diagnostics,
            "residualized": residualized,
            "solver_residual_max": solver_residual,
            "solver_relative_residual_max": solver_residual / max(
                float(torch.cat((self.statistics.Q_p, residual.Q_r), dim=0).abs().max().item()),
                1.0,
            ),
        }

    def update(self, raw_features: torch.Tensor, labels: torch.Tensor) -> None:
        anchor_features = self._anchor_features(raw_features)
        self.update_from_views(raw_features, anchor_features, labels)

    def update_from_views(
        self,
        raw_features: torch.Tensor,
        anchor_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Update using a verified fixed-anchor cache; cache is not learner state."""
        self.statistics.update(raw_features, anchor_features, labels)
        self._recompute()
        self.assert_exemplar_free_state()

    def predict_logits(self, raw_features: torch.Tensor) -> torch.Tensor:
        if self.anchor_classifier is None:
            raise RuntimeError("update() must be called before prediction")
        phi = self._anchor_features(raw_features)
        return self.predict_logits_from_views(raw_features, phi)

    def predict_logits_from_views(
        self,
        raw_features: torch.Tensor,
        anchor_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score cached fixed views without retaining either sample matrix."""
        if self.anchor_classifier is None:
            raise RuntimeError("update() must be called before prediction")
        x = raw_features.to(device=self.device, dtype=self.dtype)
        phi = anchor_features.to(device=self.device, dtype=self.dtype)
        if x.ndim != 2 or x.shape[1] != self.raw_dim:
            raise ValueError(f"features must have shape (B, {self.raw_dim})")
        if phi.shape != (x.shape[0], self.anchor_dim):
            raise ValueError(f"anchor_features must have shape (B, {self.anchor_dim})")
        if not bool(torch.isfinite(x).all() and torch.isfinite(phi).all()):
            raise ValueError("features contain NaN or Inf")
        logits = phi @ self.anchor_classifier
        if self.directions is not None:
            residual = x @ self.directions - phi @ self.complement
            logits = logits + residual @ self.residual_classifier
        return logits

    def restore_sufficient_statistics(
        self,
        statistics_state: dict,
        anchor_projection: torch.Tensor,
    ) -> None:
        """Restore a cumulative statistic snapshot and rebuild derived state."""
        projection = anchor_projection.to(self.device)
        if projection.shape != (self.anchor_dim, self.raw_dim):
            raise ValueError("invalid anchor projection shape")
        if projection.layout != torch.sparse_csc:
            raise ValueError("anchor projection must use sparse CSC layout")
        if not bool(torch.isfinite(projection.values()).all()):
            raise ValueError("anchor projection contains NaN or Inf")
        self.anchor.projection_matrix = projection
        self.statistics.load_state_dict(statistics_state)
        self._recompute()
        self.assert_exemplar_free_state()

    def predict(self, raw_features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(raw_features).argmax(dim=1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns], dtype=torch.long)

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "anchor_projection": self.anchor.projection_matrix,
            "G_pp": self.statistics.G_pp,
            "G_xx": self.statistics.G_xx,
            "H_px": self.statistics.H_px,
            "Q_p": self.statistics.Q_p,
            "Q_x": self.statistics.Q_x,
            "counts": self.statistics.counts,
        }
        for name in ("anchor_classifier", "directions", "complement", "residual_classifier"):
            tensor = getattr(self, name)
            if tensor is not None:
                tensors[name] = tensor
        return tensors

    def persistent_state_bytes(self) -> int:
        return sum(_tensor_bytes(tensor) for tensor in self.persistent_tensors().values())

    def assert_exemplar_free_state(self) -> None:
        forbidden = ("dataset", "loader", "cache", "memory", "historical", "sample", "image", "replay", "batch")
        names = tuple(self.__dict__) + tuple(self.statistics.__dict__)
        offending = [name for name in names if any(token in name.lower() for token in forbidden)]
        if offending:
            raise AssertionError(f"forbidden sample-level state names: {offending}")
        total = self.statistics.total_count
        if total > max(self.raw_dim, self.anchor_dim, self.statistics.num_classes):
            for name, tensor in self.persistent_tensors().items():
                if tensor.ndim and total in tensor.shape:
                    raise AssertionError(f"{name} has a historical sample-count dimension")

    def state_dict(self) -> dict:
        """Store fixed anchor/config/statistics only; rebuild derived tensors."""
        return {
            "version": 1,
            "method": self.method,
            "raw_dim": self.raw_dim,
            "anchor_dim": self.anchor_dim,
            "synaptic_degree": self.synaptic_degree,
            "coding_level": self.coding_level,
            "anchor_ridge": self.anchor_ridge,
            "residual_ridge": self.residual_ridge,
            "complement_ridge": self.complement_ridge,
            "requested_rank": self.requested_rank,
            "confusion_temperature": self.confusion_temperature,
            "scatter_epsilon": self.scatter_epsilon,
            "seed": self.seed,
            "anchor_projection": self.anchor.projection_matrix,
            "statistics": self.statistics.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported CRT-SOHO checkpoint version")
        fields = (
            "method", "raw_dim", "anchor_dim", "synaptic_degree", "coding_level",
            "anchor_ridge", "residual_ridge", "complement_ridge", "requested_rank",
            "confusion_temperature", "scatter_epsilon", "seed",
        )
        for field in fields:
            if field not in state:
                raise ValueError(f"checkpoint missing {field}")
            if state[field] != getattr(self, field):
                raise ValueError(f"checkpoint configuration mismatch for {field}")
        projection = state["anchor_projection"].to(self.device)
        if projection.shape != (self.anchor_dim, self.raw_dim):
            raise ValueError("invalid anchor projection shape")
        if projection.layout != torch.sparse_csc:
            raise ValueError("anchor projection must use sparse CSC layout")
        if not bool(torch.isfinite(projection.values()).all()):
            raise ValueError("anchor projection contains NaN or Inf")
        self.anchor.projection_matrix = projection
        self.statistics.load_state_dict(state["statistics"])
        self._recompute()
        self.assert_exemplar_free_state()


def create_learner(**kwargs) -> CRTSOHOLearner:
    return CRTSOHOLearner(**kwargs)


assert "task_id" not in inspect.signature(CRTSOHOLearner.predict_logits).parameters
