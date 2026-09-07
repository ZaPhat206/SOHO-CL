"""FLY representation wrapper for a generic analytic Ridge backend."""

from __future__ import annotations

import torch

from methods.analytic_ridge import AnalyticRidgeBackend, persistent_tensor_bytes
from models.flyhash import FlyHash


class FLYAnalyticLearner:
    """Compose the frozen FLY map with a representation-agnostic backend."""

    is_exemplar_free = True

    def __init__(
        self,
        *,
        feature_dim: int,
        synaptic_degree: int,
        coding_level: float,
        backend: AnalyticRidgeBackend,
        seed: int = 2025,
        projection: torch.Tensor | None = None,
    ) -> None:
        if feature_dim <= 0 or synaptic_degree <= 0:
            raise ValueError("FLY dimensions must be positive")
        if synaptic_degree > feature_dim or not 0 < coding_level <= 1:
            raise ValueError("invalid FLY representation")
        self.feature_dim = int(feature_dim)
        self.expand_dim = backend.dimension
        self.synaptic_degree = int(synaptic_degree)
        self.coding_level = float(coding_level)
        self.seed = int(seed)
        self.backend = backend
        self.device = backend.device
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.flyhash = FlyHash(
                self.feature_dim, self.expand_dim, self.synaptic_degree
            ).to(self.device)
        if projection is not None:
            supplied = projection.to(self.device)
            if supplied.shape != (self.expand_dim, self.feature_dim):
                raise ValueError("projection shape mismatch")
            finite = supplied.values() if supplied.layout == torch.sparse_csc else supplied
            if not bool(torch.isfinite(finite).all()):
                raise ValueError("projection contains NaN or Inf")
            self.flyhash.projection_matrix = supplied
        if self.flyhash.projection_matrix.layout != torch.sparse_csc:
            self.flyhash.to_sparse()

    @property
    def class_ids(self) -> list[int]:
        return self.backend.class_ids

    @property
    def weights(self) -> torch.Tensor | None:
        return self.backend.weights

    @property
    def Q(self) -> torch.Tensor:
        return self.backend.Q

    @property
    def counts(self) -> torch.Tensor:
        return self.backend.counts

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        values = features.to(
            device=self.device, dtype=self.flyhash.projection_matrix.dtype
        )
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B, {self.feature_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        return self.flyhash(values, self.coding_level, absolute_wta=False).to(
            self.backend.statistics_dtype
        )

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.backend.update(self.encode(features), labels)

    def update_codes(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        self.backend.update(codes, labels)

    def predict_logits_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.backend.predict_logits(codes)

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.predict_logits_from_codes(self.encode(features))

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(features).argmax(1).cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns])

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {"projection": self.flyhash.projection_matrix}
        tensors.update(self.backend.persistent_tensors())
        return tensors

    def persistent_state_bytes(self) -> int:
        return persistent_tensor_bytes(self.persistent_tensors())

    def assert_exemplar_free_state(self) -> None:
        self.backend.assert_exemplar_free_state()
        forbidden = ("history", "sample", "feature_cache", "labels", "codes")
        for name in self.persistent_tensors():
            if any(token in name.lower() for token in forbidden):
                raise AssertionError(f"forbidden sample-level state: {name}")

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "frontend": "fly",
            "feature_dim": self.feature_dim,
            "expand_dim": self.expand_dim,
            "synaptic_degree": self.synaptic_degree,
            "coding_level": self.coding_level,
            "seed": self.seed,
            "projection": self.flyhash.projection_matrix.detach().cpu(),
            "backend": self.backend.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1 or state.get("frontend") != "fly":
            raise ValueError("unsupported FLY frontend checkpoint")
        expected = {
            "feature_dim": self.feature_dim,
            "expand_dim": self.expand_dim,
            "synaptic_degree": self.synaptic_degree,
            "coding_level": self.coding_level,
            "seed": self.seed,
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(f"frontend checkpoint mismatch for {field}")
        projection = state["projection"].to(self.device)
        if projection.layout != torch.sparse_csc or projection.shape != (
            self.expand_dim,
            self.feature_dim,
        ):
            raise ValueError("invalid frontend projection")
        if not bool(torch.isfinite(projection.values()).all()):
            raise ValueError("frontend projection contains NaN or Inf")
        self.flyhash.projection_matrix = projection
        self.backend.load_state_dict(state["backend"])

    def load_legacy_srq_fly_state_dict(self, state: dict) -> None:
        """Import a version-1 ``SquareRootFLYLearner`` checkpoint."""
        if not hasattr(self.backend, "load_legacy_srq_fly_state_dict"):
            raise TypeError("backend cannot import a legacy SRQ-FLY checkpoint")
        expected = {
            "feature_dim": self.feature_dim,
            "expand_dim": self.expand_dim,
            "synaptic_degree": self.synaptic_degree,
            "coding_level": self.coding_level,
            "seed": self.seed,
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(f"legacy frontend mismatch for {field}")
        projection = state["projection"].to(self.device)
        if projection.layout != torch.sparse_csc:
            raise ValueError("legacy projection must be sparse CSC")
        self.flyhash.projection_matrix = projection
        self.backend.load_legacy_srq_fly_state_dict(state)


__all__ = ["FLYAnalyticLearner"]
