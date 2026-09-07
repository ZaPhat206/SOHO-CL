"""Dense random-ReLU frontend for the additive analytic Ridge backends."""

from __future__ import annotations

import torch

from methods.analytic_ridge import AnalyticRidgeBackend, persistent_tensor_bytes


class RanPACAnalyticLearner:
    """RanPAC Phase-2 map composed with a representation-agnostic backend.

    This class implements the no-PETL analytic path ``relu(F @ W_rand)`` from
    RanPAC. Ridge selection belongs to the experiment protocol, not this
    frontend, and the backend owns all additive sufficient statistics.
    """

    is_exemplar_free = True

    def __init__(
        self,
        *,
        feature_dim: int,
        backend: AnalyticRidgeBackend,
        seed: int = 2025,
        projection: torch.Tensor | None = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.expand_dim = backend.dimension
        self.seed = int(seed)
        self.backend = backend
        self.device = backend.device
        if projection is None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            projection = torch.randn(
                self.feature_dim,
                self.expand_dim,
                generator=generator,
                dtype=self.backend.statistics_dtype,
            )
        supplied = projection.to(
            device=self.device, dtype=self.backend.statistics_dtype
        )
        if supplied.layout != torch.strided or supplied.shape != (
            self.feature_dim,
            self.expand_dim,
        ):
            raise ValueError("RanPAC projection shape/layout mismatch")
        if not bool(torch.isfinite(supplied).all()):
            raise ValueError("RanPAC projection contains NaN or Inf")
        self.projection = supplied

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
            device=self.device, dtype=self.backend.statistics_dtype
        )
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B, {self.feature_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        return torch.relu(values @ self.projection)

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.backend.update(self.encode(features), labels)

    def update_codes(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        self.backend.update(codes, labels)

    def predict_logits_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.backend.predict_logits(codes)

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.predict_logits_from_codes(self.encode(features))

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.backend.predict(self.encode(features))

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {"projection": self.projection}
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
            "frontend": "ranpac_random_relu",
            "feature_dim": self.feature_dim,
            "expand_dim": self.expand_dim,
            "seed": self.seed,
            "projection": self.projection.detach().cpu().clone(),
            "backend": self.backend.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1 or state.get("frontend") != "ranpac_random_relu":
            raise ValueError("unsupported RanPAC frontend checkpoint")
        for field, expected in (
            ("feature_dim", self.feature_dim),
            ("expand_dim", self.expand_dim),
            ("seed", self.seed),
        ):
            if state.get(field) != expected:
                raise ValueError(f"RanPAC checkpoint mismatch for {field}")
        projection = state["projection"].to(
            device=self.device, dtype=self.backend.statistics_dtype
        )
        if projection.layout != torch.strided or projection.shape != (
            self.feature_dim,
            self.expand_dim,
        ):
            raise ValueError("invalid RanPAC checkpoint projection")
        if not bool(torch.isfinite(projection).all()):
            raise ValueError("RanPAC checkpoint projection contains NaN or Inf")
        self.projection = projection
        self.backend.load_state_dict(state["backend"])
        self.assert_exemplar_free_state()


__all__ = ["RanPACAnalyticLearner"]
