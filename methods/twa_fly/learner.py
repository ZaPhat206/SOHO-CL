"""Exemplar-free Two-Way Analytic FLY learner."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from methods.flycl import select_ridge_parameter
from models.flyhash import FlyHash

from .solver import solve_one_way, solve_symmetric
from .statistics import TWAStatistics


TWA_METHODS = {"twa_one_way", "twa_symmetric", "twa_shuffled_cross"}


class _RestoredFlyHash(nn.Module):
    forward = FlyHash.forward

    def __init__(self, projection: torch.Tensor, raw_dim: int, fly_dim: int, synaptic_degree: int):
        super().__init__()
        self.in_dim = raw_dim
        self.expand_dim = fly_dim
        self.synaptic_degree = synaptic_degree
        self.register_buffer("projection_matrix", projection)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    if tensor.layout == torch.sparse_csc:
        return sum(part.numel() * part.element_size() for part in (
            tensor.ccol_indices(), tensor.row_indices(), tensor.values()
        ))
    raise ValueError(f"unsupported persistent tensor layout: {tensor.layout}")


class TWAFLYLearner:
    """Paired raw/WTA streaming Ridge with WTA-only global inference."""

    is_exemplar_free = True

    def __init__(
        self,
        *,
        method: str,
        raw_dim: int,
        fly_dim: int,
        num_classes: int,
        synaptic_degree: int,
        coding_level: float,
        rho: float,
        raw_ridge: float,
        fly_ridge: float | None = None,
        ridge_lower: float = 6,
        ridge_upper: float = 10,
        solver_tolerance: float = 1e-5,
        solver_max_iterations: int = 100,
        seed: int = 1993,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        projection: torch.Tensor | None = None,
    ):
        if method not in TWA_METHODS:
            raise ValueError(f"unknown TWA-FLY method {method!r}; choices: {sorted(TWA_METHODS)}")
        if min(raw_dim, fly_dim, num_classes) <= 0:
            raise ValueError("feature dimensions and num_classes must be positive")
        if not 0 < synaptic_degree <= raw_dim:
            raise ValueError("synaptic_degree must be in [1, raw_dim]")
        if not 0 < coding_level <= 1 or int(fly_dim * coding_level) < 1:
            raise ValueError("coding_level must retain at least one coordinate")
        if rho < 0 or raw_ridge <= 0 or (fly_ridge is not None and fly_ridge <= 0):
            raise ValueError("rho must be non-negative and Ridge coefficients positive")
        if fly_ridge is None and ridge_lower >= ridge_upper:
            raise ValueError("ridge_lower must be smaller than ridge_upper for GCV")
        if solver_tolerance <= 0 or solver_max_iterations <= 0:
            raise ValueError("solver_tolerance and solver_max_iterations must be positive")
        self.method = method
        self.raw_dim = int(raw_dim)
        self.fly_dim = int(fly_dim)
        self.num_classes = int(num_classes)
        self.synaptic_degree = int(synaptic_degree)
        self.coding_level = float(coding_level)
        self.rho = float(rho)
        self.raw_ridge = float(raw_ridge)
        self.fly_ridge = None if fly_ridge is None else float(fly_ridge)
        self.ridge_lower = float(ridge_lower)
        self.ridge_upper = float(ridge_upper)
        self.solver_tolerance = float(solver_tolerance)
        self.solver_max_iterations = int(solver_max_iterations)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype
        if projection is None:
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(self.seed)
                flyhash = FlyHash(self.raw_dim, self.fly_dim, self.synaptic_degree).to(self.device)
            flyhash.to_sparse()
        else:
            if projection.layout != torch.sparse_csc or projection.shape != (self.fly_dim, self.raw_dim):
                raise ValueError("projection must be sparse CSC with shape (fly_dim, raw_dim)")
            if not bool(torch.isfinite(projection.values()).all()):
                raise ValueError("projection contains NaN or Inf")
            flyhash = _RestoredFlyHash(
                projection.to(self.device), self.raw_dim, self.fly_dim, self.synaptic_degree
            )
        self.flyhash = flyhash
        self.statistics = TWAStatistics(
            self.raw_dim, self.fly_dim, self.num_classes, device=self.device, dtype=self.dtype
        )
        self.raw_weights: torch.Tensor | None = None
        self.fly_weights: torch.Tensor | None = None
        self.last_fly_ridge: float | None = None
        self.updates_seen = 0
        self.diagnostics: dict = {
            "method": self.method,
            "projection": "fixed_sparse_gaussian_wta",
            "ridge_policy": "fixed" if self.fly_ridge is not None else "original_current_task_gcv",
            "inference_view": "fly_wta_only",
        }

    @property
    def class_ids(self) -> list[int]:
        return self.statistics.class_ids

    def encode_fly(self, raw_features: torch.Tensor) -> torch.Tensor:
        x = raw_features.to(device=self.device, dtype=self.flyhash.projection_matrix.dtype)
        if x.ndim != 2 or x.shape[1] != self.raw_dim:
            raise ValueError(f"features must have shape (B, {self.raw_dim})")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("features contain NaN or Inf")
        return self.flyhash(x, self.coding_level, absolute_wta=False).to(self.dtype)

    def encode_sparse_fly(self, raw_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact active indices/values without retaining dense rows."""
        x = raw_features.to(device=self.device, dtype=self.flyhash.projection_matrix.dtype)
        if x.ndim != 2 or x.shape[1] != self.raw_dim:
            raise ValueError(f"features must have shape (B, {self.raw_dim})")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("features contain NaN or Inf")
        projection = self.flyhash.projection_matrix
        projected = torch.sparse.mm(projection, x.T) if projection.layout == torch.sparse_csc else projection @ x.T
        values, indices = projected.topk(
            max(1, int(self.fly_dim * self.coding_level)), dim=0, largest=True
        )
        return indices.T, values.T.to(self.dtype)

    def _targets(self, labels: torch.Tensor) -> torch.Tensor:
        y = labels.to(self.device, torch.long)
        if y.ndim != 1 or bool(((y < 0) | (y >= self.num_classes)).any()):
            raise ValueError("labels must be global class IDs in [0, num_classes)")
        return torch.nn.functional.one_hot(y, self.num_classes).to(self.dtype)

    def update(self, raw_features: torch.Tensor, labels: torch.Tensor) -> None:
        self.update_from_encoded(raw_features, self.encode_fly(raw_features), labels)

    def update_from_encoded(
        self, raw_features: torch.Tensor, fly_features: torch.Tensor, labels: torch.Tensor
    ) -> None:
        x = raw_features.to(device=self.device, dtype=self.dtype)
        z = fly_features.to(device=self.device, dtype=self.dtype)
        y = labels.to(device=self.device, dtype=torch.long)
        targets = self._targets(y)
        selected = self.fly_ridge
        if selected is None:
            selected = float(select_ridge_parameter(
                z, targets, self.ridge_lower, self.ridge_upper
            ).item())
        cross_z = None
        if self.method == "twa_shuffled_cross":
            generator = torch.Generator(device="cpu").manual_seed(
                self.seed + 104729 + self.updates_seen
            )
            permutation = torch.randperm(z.shape[0], generator=generator).to(self.device)
            cross_z = z[permutation]
        self.statistics.update(x, z, y, cross_fly_features=cross_z)
        self.updates_seen += 1
        self.last_fly_ridge = float(selected)
        cross = self.statistics.R_xz
        if self.method == "twa_one_way":
            solution = solve_one_way(
                self.statistics, self.rho, self.raw_ridge, self.last_fly_ridge, cross=cross
            )
        else:
            solution = solve_symmetric(
                self.statistics,
                self.rho,
                self.raw_ridge,
                self.last_fly_ridge,
                tolerance=self.solver_tolerance,
                max_iterations=self.solver_max_iterations,
                cross=cross,
            )
        self.raw_weights = solution.raw_weights
        self.fly_weights = solution.fly_weights
        self.diagnostics.update(
            selected_ridge=self.last_fly_ridge,
            solver_relative_residual=solution.relative_residual,
            solver_iterations=solution.iterations,
            objective_history=list(solution.objective_history),
            total_count=self.statistics.total_count,
        )

    def predict_logits(self, raw_features: torch.Tensor) -> torch.Tensor:
        if self.fly_weights is None:
            raise RuntimeError("update() must be called before prediction")
        return self.encode_fly(raw_features) @ self.fly_weights

    def predict_logits_from_encoded(self, fly_features: torch.Tensor) -> torch.Tensor:
        if self.fly_weights is None:
            raise RuntimeError("update() must be called before prediction")
        z = fly_features.to(device=self.device, dtype=self.dtype)
        if z.ndim != 2 or z.shape[1] != self.fly_dim:
            raise ValueError(f"fly_features must have shape (B, {self.fly_dim})")
        return z @ self.fly_weights

    def predict(self, raw_features: torch.Tensor) -> torch.Tensor:
        return self.predict_logits(raw_features).argmax(1).detach().cpu()

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "projection": self.flyhash.projection_matrix,
            **{name: getattr(self.statistics, name) for name in (
                "G_xx", "G_zz", "R_xz", "Q_x", "Q_z", "counts"
            )},
        }
        if self.raw_weights is not None:
            tensors["raw_weights"] = self.raw_weights
        if self.fly_weights is not None:
            tensors["fly_weights"] = self.fly_weights
        return tensors

    def persistent_state_bytes(self) -> int:
        return sum(_tensor_bytes(tensor) for tensor in self.persistent_tensors().values())

    def assert_exemplar_free_state(self) -> None:
        if not self.is_exemplar_free:
            raise AssertionError("learner is not marked exemplar-free")
        allowed = {
            (self.raw_dim, self.raw_dim), (self.fly_dim, self.fly_dim),
            (self.raw_dim, self.fly_dim), (self.raw_dim, self.num_classes),
            (self.fly_dim, self.num_classes), (self.num_classes,),
            (self.fly_dim, self.raw_dim),
        }
        for name, tensor in self.persistent_tensors().items():
            if tuple(tensor.shape) not in allowed:
                raise AssertionError(f"unexpected persistent tensor shape for {name}: {tuple(tensor.shape)}")
        if "task_id" in inspect.signature(self.predict_logits).parameters:
            raise AssertionError("inference must not accept task_id")

    def state_dict(self) -> dict:
        return {
            "version": 1,
            **{name: getattr(self, name) for name in (
                "method", "raw_dim", "fly_dim", "num_classes", "synaptic_degree",
                "coding_level", "rho", "raw_ridge", "fly_ridge", "ridge_lower",
                "ridge_upper", "solver_tolerance", "solver_max_iterations", "seed"
            )},
            "projection": self.flyhash.projection_matrix.detach().cpu(),
            "statistics": self.statistics.state_dict(),
            "last_fly_ridge": self.last_fly_ridge,
            "updates_seen": self.updates_seen,
        }

    def load_state_dict(self, state: dict) -> None:
        keys = (
            "method", "raw_dim", "fly_dim", "num_classes", "synaptic_degree",
            "coding_level", "rho", "raw_ridge", "fly_ridge", "ridge_lower",
            "ridge_upper", "solver_tolerance", "solver_max_iterations", "seed",
        )
        mismatches = [name for name in keys if state[name] != getattr(self, name)]
        if mismatches:
            raise ValueError(f"TWA-FLY checkpoint configuration mismatch: {mismatches}")
        projection = state["projection"]
        if projection.layout != torch.sparse_csc or projection.shape != (self.fly_dim, self.raw_dim):
            raise ValueError("checkpoint projection is incompatible")
        self.flyhash = _RestoredFlyHash(
            projection.to(self.device), self.raw_dim, self.fly_dim, self.synaptic_degree
        )
        self.statistics.load_state_dict(state["statistics"])
        self.last_fly_ridge = None if state["last_fly_ridge"] is None else float(state["last_fly_ridge"])
        self.updates_seen = int(state["updates_seen"])
        if self.updates_seen < 0:
            raise ValueError("updates_seen must be non-negative")
        if self.last_fly_ridge is None:
            self.raw_weights = self.fly_weights = None
            return
        cross = self.statistics.R_xz
        solution = solve_one_way(
            self.statistics, self.rho, self.raw_ridge, self.last_fly_ridge, cross=cross
        ) if self.method == "twa_one_way" else solve_symmetric(
            self.statistics, self.rho, self.raw_ridge, self.last_fly_ridge,
            tolerance=self.solver_tolerance, max_iterations=self.solver_max_iterations,
            cross=cross,
        )
        self.raw_weights, self.fly_weights = solution.raw_weights, solution.fly_weights
        self.diagnostics.update(
            selected_ridge=self.last_fly_ridge,
            solver_relative_residual=solution.relative_residual,
            solver_iterations=solution.iterations,
            objective_history=list(solution.objective_history),
            total_count=self.statistics.total_count,
        )
