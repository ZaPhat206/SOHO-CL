"""Replay-free learner for Moment-Transport SOHO."""

from __future__ import annotations

import inspect

import torch

from models.flyhash import FlyHash

from .geometry import class_geometry, shuffled_targets, solve_spd, transport_moments
from .statistics import MomentTransportStatistics


METHODS = {
    "fixed_wta_ridge",
    "mt_unwhitened",
    "mt_whitened",
    "mt_shuffled",
}


def _tensor_storage_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    if tensor.layout == torch.sparse_csc:
        return (
            tensor.values().numel() * tensor.values().element_size()
            + tensor.ccol_indices().numel() * tensor.ccol_indices().element_size()
            + tensor.row_indices().numel() * tensor.row_indices().element_size()
        )
    raise ValueError(f"unsupported persistent tensor layout: {tensor.layout}")


class MTSOHOLearner:
    """Fixed nonlinear anchors with sample-independent analytic adaptation.

    No attribute stores a historical feature, label, WTA code, or sample index.
    The projection is fixed.  Every changing representation is linear *after*
    WTA, so its full-stream Gram and cross matrices are transported exactly.
    """

    is_exemplar_free = True

    def __init__(
        self,
        *,
        method: str,
        feature_dim: int,
        expand_dim: int,
        synaptic_degree: int,
        coding_level: float,
        anchor_ridge: float,
        projection_ridge: float = 10.0,
        adapted_ridge: float = 10.0,
        target_rank: int = 32,
        shrinkage: float = 0.1,
        adaptation_weight: float = 1.0,
        geometry_epsilon: float = 1e-5,
        seed: int = 2025,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if method not in METHODS:
            raise ValueError(f"unknown MT-SOHO method {method!r}")
        if feature_dim <= 0 or expand_dim <= 0 or not 0 < synaptic_degree <= feature_dim:
            raise ValueError("invalid feature/projection dimensions")
        if not 0 < coding_level <= 1:
            raise ValueError("coding_level must be in (0, 1]")
        if min(anchor_ridge, projection_ridge, adapted_ridge) <= 0:
            raise ValueError("all Ridge coefficients must be positive")
        if target_rank <= 0 or adaptation_weight < 0:
            raise ValueError("invalid target rank or adaptation weight")

        self.method = method
        self.feature_dim, self.expand_dim = int(feature_dim), int(expand_dim)
        self.synaptic_degree, self.coding_level = int(synaptic_degree), float(coding_level)
        self.anchor_ridge = float(anchor_ridge)
        self.projection_ridge, self.adapted_ridge = float(projection_ridge), float(adapted_ridge)
        self.target_rank, self.shrinkage = int(target_rank), float(shrinkage)
        self.adaptation_weight, self.geometry_epsilon = float(adaptation_weight), float(geometry_epsilon)
        self.seed = int(seed)
        self.device, self.dtype = torch.device(device), dtype

        devices = [] if self.device.type == "cpu" else [
            self.device.index if self.device.index is not None else torch.cuda.current_device()
        ]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)
            self.flyhash = FlyHash(self.feature_dim, self.expand_dim, self.synaptic_degree).to(
                device=self.device, dtype=self.dtype
            )
        self.flyhash.to_sparse()
        self.statistics = MomentTransportStatistics(
            self.feature_dim, self.expand_dim, device=self.device, dtype=self.dtype
        )
        self.anchor_weights: torch.Tensor | None = None
        self.transport: torch.Tensor | None = None
        self.adapted_weights: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None
        self.diagnostics: dict = {"representation": "fixed_wta_anchor"}

    @property
    def class_ids(self) -> list[int]:
        return self.statistics.class_ids

    def _encode(self, features: torch.Tensor) -> torch.Tensor:
        values = features.to(self.device, self.dtype)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B, {self.feature_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        return self.flyhash(values, self.coding_level, absolute_wta=False)

    def _recompute(self) -> None:
        stats = self.statistics
        eye = torch.eye(self.expand_dim, device=self.device, dtype=self.dtype)
        anchor_system = stats.G_u + self.anchor_ridge * eye
        self.anchor_weights = solve_spd(anchor_system, stats.Q_u)
        self.transport = self.adapted_weights = self.targets = None
        self.diagnostics = {
            "representation": "fixed_wta_anchor",
            "seen_classes": stats.num_classes,
        }
        if self.method == "fixed_wta_ridge" or stats.num_classes < 2:
            return

        targets, geometry = class_geometry(
            raw_gram=stats.G_x,
            raw_cross=stats.Q_x,
            counts=stats.counts,
            requested_rank=self.target_rank,
            shrinkage=self.shrinkage,
            epsilon=self.geometry_epsilon,
            whiten=self.method != "mt_unwhitened",
        )
        if self.method == "mt_shuffled":
            targets = shuffled_targets(targets, self.seed + stats.num_classes)
        projection_rhs = stats.Q_u @ targets
        self.transport = solve_spd(
            stats.G_u + self.projection_ridge * eye, projection_rhs
        )
        gram_v, cross_v = transport_moments(stats.G_u, stats.Q_u, self.transport)
        adapted_eye = torch.eye(gram_v.shape[0], device=self.device, dtype=self.dtype)
        self.adapted_weights = solve_spd(
            gram_v + self.adapted_ridge * adapted_eye, cross_v
        )
        self.targets = targets
        residual = (
            (gram_v + self.adapted_ridge * adapted_eye) @ self.adapted_weights
            - cross_v
        )
        self.diagnostics.update(
            adaptation="post_wta_exact_moment_transport",
            shuffled=self.method == "mt_shuffled",
            solver_relative_residual=float(
                residual.norm().div(cross_v.norm().clamp_min(self.geometry_epsilon)).item()
            ),
            **geometry,
        )

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        raw = features.to(self.device, self.dtype)
        anchor = self._encode(raw)
        self.statistics.update(raw, anchor, labels)
        self._recompute()
        self.assert_exemplar_free_state()

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.anchor_weights is None:
            raise RuntimeError("update() must be called before prediction")
        anchor = self._encode(features)
        logits = anchor @ self.anchor_weights
        if self.transport is not None and self.adapted_weights is not None:
            logits = logits + self.adaptation_weight * (anchor @ self.transport @ self.adapted_weights)
        return logits

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(features).argmax(1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns], dtype=torch.long)

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "projection": self.flyhash.projection_matrix,
            "anchor_gram": self.statistics.G_u,
            "raw_gram": self.statistics.G_x,
            "anchor_class_cross": self.statistics.Q_u,
            "raw_class_cross": self.statistics.Q_x,
            "class_counts": self.statistics.counts,
        }
        for name in ("anchor_weights", "transport", "adapted_weights", "targets"):
            value = getattr(self, name)
            if value is not None:
                tensors[name] = value
        return tensors

    def persistent_state_bytes(self) -> int:
        return sum(_tensor_storage_bytes(value) for value in self.persistent_tensors().values())

    def assert_exemplar_free_state(self) -> None:
        forbidden = ("history", "replay", "sample", "image", "label_buffer", "feature_buffer")
        names = tuple(self.__dict__) + tuple(self.statistics.__dict__)
        offending = [name for name in names if any(token in name.lower() for token in forbidden)]
        if offending:
            raise AssertionError(f"forbidden sample-level state names: {offending}")
        total = self.statistics.total_count
        if total > max(self.feature_dim, self.expand_dim, self.statistics.num_classes):
            for name, value in self.persistent_tensors().items():
                if value.ndim and total in value.shape:
                    raise AssertionError(f"{name} has historical sample-count dimension")

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "config": {
                "method": self.method,
                "feature_dim": self.feature_dim,
                "expand_dim": self.expand_dim,
                "synaptic_degree": self.synaptic_degree,
                "coding_level": self.coding_level,
                "anchor_ridge": self.anchor_ridge,
                "projection_ridge": self.projection_ridge,
                "adapted_ridge": self.adapted_ridge,
                "target_rank": self.target_rank,
                "shrinkage": self.shrinkage,
                "adaptation_weight": self.adaptation_weight,
                "geometry_epsilon": self.geometry_epsilon,
                "seed": self.seed,
            },
            "statistics": self.statistics.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported MT-SOHO checkpoint version")
        expected = self.state_dict()["config"]
        if state.get("config") != expected:
            raise ValueError("MT-SOHO checkpoint configuration mismatch")
        self.statistics.load_state_dict(state["statistics"])
        self._recompute()
        self.assert_exemplar_free_state()


assert "task_id" not in inspect.signature(MTSOHOLearner.predict_logits).parameters
