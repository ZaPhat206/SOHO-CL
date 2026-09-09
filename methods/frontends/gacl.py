"""Controlled GACL frontend and inverse-RLS equation reference.

The frontend mirrors the random bias-free linear layer followed by ReLU used
by GACL. The reference implements the paper's Woodbury recurrence, while the
generic learner delegates the equivalent primal additive-Ridge system to the
shared analytic backend.
"""

from __future__ import annotations

import math

import torch

from methods.analytic_ridge import AnalyticRidgeBackend, persistent_tensor_bytes


def gacl_linear_projection(
    feature_dim: int,
    expansion_dim: int,
    *,
    seed: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    """Reproduce the default weight initialization of a bias-free Linear."""

    if feature_dim <= 0 or expansion_dim <= 0:
        raise ValueError("feature and expansion dimensions must be positive")
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("GACL projection dtype must be float32 or float64")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    # Upstream nn.Linear stores (out_features, in_features). Generate that
    # shape before transposing so a fixed seed has the same random ordering.
    weight = torch.empty((expansion_dim, feature_dim), dtype=dtype, device="cpu")
    bound = 1.0 / math.sqrt(feature_dim)
    weight.uniform_(-bound, bound, generator=generator)
    return weight.T.contiguous().to(device=device)


class GACLAnalyticLearner:
    """Random-ReLU GACL feature map composed with an additive-Ridge backend."""

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
        self.expansion_dim = backend.dimension
        self.seed = int(seed)
        self.backend = backend
        self.device = backend.device
        if projection is None:
            projection = gacl_linear_projection(
                self.feature_dim,
                self.expansion_dim,
                seed=self.seed,
                dtype=backend.statistics_dtype,
                device=self.device,
            )
        supplied = projection.to(
            device=self.device, dtype=self.backend.statistics_dtype
        )
        if supplied.layout != torch.strided or supplied.shape != (
            self.feature_dim,
            self.expansion_dim,
        ):
            raise ValueError("GACL projection shape/layout mismatch")
        if not bool(torch.isfinite(supplied).all()):
            raise ValueError("GACL projection contains NaN or Inf")
        self.projection = supplied

    @property
    def class_ids(self) -> list[int]:
        return self.backend.class_ids

    @property
    def weights(self) -> torch.Tensor | None:
        return self.backend.weights

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


class GACLInverseRLSReference:
    """Equation-level reference for GACL's inverse-RLS recurrence.

    This baseline intentionally keeps the inverse autocorrelation matrix. Its
    columns are sorted by class ID so they align with the generic backend.
    """

    is_exemplar_free = True

    def __init__(
        self,
        *,
        dimension: int,
        gamma: float,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if dimension <= 0 or gamma <= 0:
            raise ValueError("dimension and gamma must be positive")
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("GACL reference dtype must be float32 or float64")
        self.dimension = int(dimension)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.dtype = dtype
        self.inverse_system = torch.eye(
            self.dimension, device=self.device, dtype=self.dtype
        ) / self.gamma
        self.weights = torch.zeros(
            (self.dimension, 0), device=self.device, dtype=self.dtype
        )
        self.class_ids: list[int] = []
        self.total_rows = 0
        self.update_count = 0
        self.diagnostics: dict[str, object] = {
            "method": "gacl_inverse_rls_reference",
            "gamma": self.gamma,
            "update_formula": "woodbury_solve_then_weight_gain",
        }

    def _expanded_weights(
        self, labels: torch.Tensor
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        updated = sorted(set(self.class_ids) | set(map(int, labels.cpu().tolist())))
        old_columns = {value: index for index, value in enumerate(self.class_ids)}
        new_columns = {value: index for index, value in enumerate(updated)}
        weights = torch.zeros(
            (self.dimension, len(updated)), device=self.device, dtype=self.dtype
        )
        for class_id, old_column in old_columns.items():
            weights[:, new_columns[class_id]] = self.weights[:, old_column]
        columns = torch.tensor(
            [new_columns[int(value)] for value in labels.cpu().tolist()],
            device=self.device,
            dtype=torch.long,
        )
        targets = torch.nn.functional.one_hot(
            columns, num_classes=len(updated)
        ).to(self.dtype)
        return updated, weights, targets

    def update_codes(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        values = codes.to(device=self.device, dtype=self.dtype)
        target_labels = labels.to(device=self.device, dtype=torch.long)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"codes must have shape (B, {self.dimension})")
        if target_labels.ndim != 1 or len(target_labels) != len(values) or not len(values):
            raise ValueError("labels must align with a non-empty code matrix")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("codes contain NaN or Inf")

        class_ids, old_weights, targets = self._expanded_weights(target_labels)
        right = values @ self.inverse_system
        kernel = torch.eye(len(values), device=self.device, dtype=self.dtype)
        kernel.add_(right @ values.T)
        # This solve is algebraically the paper's Woodbury inverse term while
        # avoiding an explicit matrix inverse in the equation-level reference.
        correction = self.inverse_system @ values.T @ torch.linalg.solve(
            kernel, right
        )
        updated_inverse = self.inverse_system - correction
        residual = targets - values @ old_weights
        updated_weights = old_weights + updated_inverse @ values.T @ residual

        if not bool(torch.isfinite(updated_inverse).all()) or not bool(
            torch.isfinite(updated_weights).all()
        ):
            raise RuntimeError("GACL inverse-RLS update became non-finite")
        self.inverse_system = updated_inverse
        self.weights = updated_weights
        self.class_ids = class_ids
        self.total_rows += len(values)
        self.update_count += 1
        symmetry_error = torch.linalg.vector_norm(
            updated_inverse - updated_inverse.T
        ) / max(float(torch.linalg.vector_norm(updated_inverse).item()), 1.0)
        self.diagnostics.update(
            total_rows=self.total_rows,
            update_count=self.update_count,
            inverse_symmetry_relative_error=float(symmetry_error.item()),
        )
        self.assert_exemplar_free_state()

    def predict_logits_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        values = codes.to(device=self.device, dtype=self.dtype)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"codes must have shape (B, {self.dimension})")
        return values @ self.weights

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        return {"inverse_system": self.inverse_system, "weights": self.weights}

    def persistent_state_bytes(self) -> int:
        return persistent_tensor_bytes(self.persistent_tensors())

    def assert_exemplar_free_state(self) -> None:
        if self.inverse_system.shape != (self.dimension, self.dimension):
            raise AssertionError("invalid inverse-system shape")
        if self.weights.shape != (self.dimension, len(self.class_ids)):
            raise AssertionError("invalid reference weight shape")
        if self.class_ids != sorted(set(self.class_ids)):
            raise AssertionError("reference class IDs must be sorted and unique")
        for name, tensor in self.persistent_tensors().items():
            if not bool(torch.isfinite(tensor).all()):
                raise AssertionError(f"non-finite reference state: {name}")


__all__ = [
    "GACLAnalyticLearner",
    "GACLInverseRLSReference",
    "gacl_linear_projection",
]
