"""Exemplar-free Prototype-Protected Sketch SOHO learner."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from models.flyhash import FlyHash

from .solver import solve_compact_ridge
from .statistics import ClassProtectedStatistics


class _RestoredFlyHash(nn.Module):
    forward = FlyHash.forward

    def __init__(self, projection: torch.Tensor, raw_dim: int, anchor_dim: int, degree: int):
        super().__init__()
        self.in_dim = raw_dim
        self.expand_dim = anchor_dim
        self.synaptic_degree = degree
        self.register_buffer("projection_matrix", projection)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.sparse_csc:
        return sum(part.numel() * part.element_size() for part in (
            tensor.ccol_indices(), tensor.row_indices(), tensor.values()
        ))
    return tensor.numel() * tensor.element_size()


class PPSSOHOLearner:
    """Fixed WTA map with protected prototypes and a covariance sketch."""

    is_exemplar_free = True

    def __init__(
        self,
        *,
        raw_dim: int,
        anchor_dim: int,
        synaptic_degree: int,
        coding_level: float,
        sketch_size: int,
        ridge_lambda: float,
        gamma: float = 1.0,
        mode: str = "class_protected",
        seed: int = 1993,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        anchor_projection: torch.Tensor | None = None,
    ) -> None:
        if raw_dim <= 0 or anchor_dim <= 0:
            raise ValueError("raw_dim and anchor_dim must be positive")
        if not 0 < synaptic_degree <= raw_dim:
            raise ValueError("synaptic_degree must be in [1, raw_dim]")
        if not 0 < coding_level <= 1 or int(anchor_dim * coding_level) < 1:
            raise ValueError("coding_level must retain at least one coordinate")
        if not 0 < sketch_size <= anchor_dim:
            raise ValueError("sketch_size must be in [1, anchor_dim]")
        if ridge_lambda <= 0 or gamma < 0:
            raise ValueError("ridge_lambda must be positive and gamma non-negative")
        self.raw_dim = int(raw_dim)
        self.anchor_dim = int(anchor_dim)
        self.synaptic_degree = int(synaptic_degree)
        self.coding_level = float(coding_level)
        self.sketch_size = int(sketch_size)
        self.ridge_lambda = float(ridge_lambda)
        self.gamma = float(gamma)
        self.mode = mode
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype

        if anchor_projection is None:
            fork_devices: list[int] = []
            if self.device.type == "cuda":
                fork_devices = [self.device.index or torch.cuda.current_device()]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(self.seed)
                anchor = FlyHash(self.raw_dim, self.anchor_dim, self.synaptic_degree)
            anchor.to_sparse()
        else:
            if anchor_projection.shape != (self.anchor_dim, self.raw_dim):
                raise ValueError("invalid anchor projection shape")
            if anchor_projection.layout != torch.sparse_csc:
                raise ValueError("anchor projection must use sparse CSC layout")
            anchor = _RestoredFlyHash(
                anchor_projection, self.raw_dim, self.anchor_dim, self.synaptic_degree
            )
        self.anchor = anchor.to(self.device)
        self.statistics = ClassProtectedStatistics(
            self.anchor_dim,
            self.sketch_size,
            mode=self.mode,
            device=self.device,
            dtype=self.dtype,
        )
        self.classifier: torch.Tensor | None = None
        self.diagnostics: dict = {
            "geometry": self.mode,
            "sketch_size": self.sketch_size,
            "gamma": self.gamma,
        }

    @property
    def class_ids(self) -> list[int]:
        return self.statistics.class_ids

    def encode(self, raw_features: torch.Tensor) -> torch.Tensor:
        values = raw_features.to(device=self.device, dtype=self.anchor.projection_matrix.dtype)
        if values.ndim != 2 or values.shape[1] != self.raw_dim:
            raise ValueError(f"features must have shape (N, {self.raw_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        return self.anchor(values, self.coding_level).to(dtype=self.dtype)

    def update(self, raw_features: torch.Tensor, labels: torch.Tensor) -> None:
        self.update_from_codes(self.encode(raw_features), labels)

    def update_from_codes(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        self.statistics.update(codes, labels)
        self.classifier, relative = solve_compact_ridge(
            self.statistics, self.ridge_lambda, self.gamma
        )
        self.diagnostics = {
            "geometry": self.mode,
            "sketch_size": self.sketch_size,
            "gamma": self.gamma,
            "covariance_error_bound": self.statistics.sketch.covariance_error_bound,
            "solver_relative_residual_max": relative,
            "total_count": self.statistics.total_count,
        }
        self.assert_exemplar_free_state()

    def predict_logits(self, raw_features: torch.Tensor) -> torch.Tensor:
        return self.predict_logits_from_codes(self.encode(raw_features))

    def predict_logits_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("update() must be called before prediction")
        values = codes.to(device=self.device, dtype=self.dtype)
        if values.ndim != 2 or values.shape[1] != self.anchor_dim:
            raise ValueError(f"codes must have shape (N, {self.anchor_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("codes contain NaN or Inf")
        return values @ self.classifier

    def predict(self, raw_features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(raw_features).argmax(dim=1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns], dtype=torch.long)

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "anchor_projection": self.anchor.projection_matrix,
            "within_sketch": self.statistics.sketch.B,
            "class_means": self.statistics.means,
            "counts": self.statistics.counts,
        }
        if self.classifier is not None:
            tensors["classifier"] = self.classifier
        return tensors

    def persistent_state_bytes(self) -> int:
        return sum(_tensor_bytes(value) for value in self.persistent_tensors().values())

    def assert_exemplar_free_state(self) -> None:
        forbidden = ("dataset", "loader", "cache", "historical", "sample", "image", "replay")
        names = tuple(self.__dict__) + tuple(self.statistics.__dict__) + tuple(self.statistics.sketch.__dict__)
        offending = [name for name in names if any(token in name.lower() for token in forbidden)]
        if offending:
            raise AssertionError(f"forbidden state names: {offending}")
        count = self.statistics.total_count
        if count > max(self.anchor_dim, self.raw_dim, self.statistics.num_classes):
            for name, tensor in self.persistent_tensors().items():
                if tensor.ndim and count in tensor.shape:
                    raise AssertionError(f"{name} has a historical sample-count dimension")

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "raw_dim": self.raw_dim,
            "anchor_dim": self.anchor_dim,
            "synaptic_degree": self.synaptic_degree,
            "coding_level": self.coding_level,
            "sketch_size": self.sketch_size,
            "ridge_lambda": self.ridge_lambda,
            "gamma": self.gamma,
            "mode": self.mode,
            "seed": self.seed,
            "anchor_projection": self.anchor.projection_matrix,
            "statistics": self.statistics.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported PPS-SOHO checkpoint version")
        for field in (
            "raw_dim", "anchor_dim", "synaptic_degree", "coding_level", "sketch_size",
            "ridge_lambda", "gamma", "mode", "seed",
        ):
            if state.get(field) != getattr(self, field):
                raise ValueError(f"checkpoint configuration mismatch for {field}")
        projection = state["anchor_projection"].to(self.device)
        if projection.shape != (self.anchor_dim, self.raw_dim):
            raise ValueError("invalid anchor projection shape")
        if projection.layout != torch.sparse_csc:
            raise ValueError("anchor projection must use sparse CSC layout")
        self.anchor.projection_matrix = projection
        self.statistics.load_state_dict(state["statistics"])
        self.classifier, relative = solve_compact_ridge(
            self.statistics, self.ridge_lambda, self.gamma
        )
        self.diagnostics = {
            "geometry": self.mode,
            "sketch_size": self.sketch_size,
            "gamma": self.gamma,
            "covariance_error_bound": self.statistics.sketch.covariance_error_bound,
            "solver_relative_residual_max": relative,
            "total_count": self.statistics.total_count,
        }
        self.assert_exemplar_free_state()


def create_learner(**kwargs) -> PPSSOHOLearner:
    return PPSSOHOLearner(**kwargs)


assert "task_id" not in inspect.signature(PPSSOHOLearner.predict_logits).parameters
