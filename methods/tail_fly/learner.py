"""Exemplar-free TAIL-FLY learner."""

from __future__ import annotations

import inspect

import torch

from models.flyhash import FlyHash

from .solver import diagonal_tail, low_rank_diagonal, solve_tail_ridge
from .streaming_svd import StreamingTruncatedSVD


def _storage_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    if tensor.layout == torch.sparse_csc:
        return (
            tensor.values().numel() * tensor.values().element_size()
            + tensor.ccol_indices().numel() * tensor.ccol_indices().element_size()
            + tensor.row_indices().numel() * tensor.row_indices().element_size()
        )
    raise ValueError(f"unsupported persistent tensor layout: {tensor.layout}")


class TAILFlyLearner:
    """Fixed full-dimensional WTA representation with compressed Gram state."""

    is_exemplar_free = True

    def __init__(
        self,
        *,
        feature_dim: int,
        expand_dim: int,
        synaptic_degree: int,
        coding_level: float,
        max_rank: int,
        ridge_lambda: float,
        seed: int = 2025,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        solver_dtype: torch.dtype | None = None,
        projection: torch.Tensor | None = None,
    ) -> None:
        if feature_dim <= 0 or expand_dim <= 0 or synaptic_degree <= 0:
            raise ValueError("representation dimensions must be positive")
        if not 0 < coding_level <= 1:
            raise ValueError("coding_level must be in (0, 1]")
        if max_rank < 0 or max_rank > expand_dim:
            raise ValueError("max_rank must be in [0, expand_dim]")
        if ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive")
        self.feature_dim = int(feature_dim)
        self.expand_dim = int(expand_dim)
        self.synaptic_degree = int(synaptic_degree)
        self.coding_level = float(coding_level)
        self.max_rank = int(max_rank)
        self.ridge_lambda = float(ridge_lambda)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype
        self.solver_dtype = dtype if solver_dtype is None else solver_dtype
        if self.solver_dtype not in {torch.float32, torch.float64}:
            raise ValueError("solver_dtype must be float32 or float64")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.flyhash = FlyHash(
                self.feature_dim, self.expand_dim, self.synaptic_degree
            ).to(device=self.device)
        if projection is not None:
            supplied = projection.to(device=self.device)
            if supplied.shape != (self.expand_dim, self.feature_dim):
                raise ValueError("projection shape mismatch")
            finite_values = (
                supplied.values() if supplied.layout == torch.sparse_csc else supplied
            )
            if not bool(torch.isfinite(finite_values).all()):
                raise ValueError("projection contains NaN or Inf")
            self.flyhash.projection_matrix = supplied
        if self.flyhash.projection_matrix.layout != torch.sparse_csc:
            self.flyhash.to_sparse()

        self.svd = StreamingTruncatedSVD(
            self.expand_dim,
            self.max_rank,
            device=self.device,
            dtype=self.dtype,
        )
        self.exact_diagonal = torch.zeros(
            self.expand_dim, device=self.device, dtype=self.dtype
        )
        self.Q = torch.zeros(
            (self.expand_dim, 0), device=self.device, dtype=self.dtype
        )
        self.counts = torch.zeros((0,), device=self.device, dtype=self.dtype)
        self.class_ids: list[int] = []
        self.weights: torch.Tensor | None = None
        self.diagnostics: dict = {
            "method": "tail_fly",
            "projection": "fixed_sparse_gaussian_wta",
            "task_id_required": False,
        }

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        values = features.to(
            device=self.device, dtype=self.flyhash.projection_matrix.dtype
        )
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B, {self.feature_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        return self.flyhash(
            values, self.coding_level, absolute_wta=False
        ).to(self.dtype)

    def _expand_classes(self, labels: torch.Tensor) -> None:
        incoming = set(map(int, labels.detach().cpu().tolist()))
        updated = sorted(set(self.class_ids) | incoming)
        if updated == self.class_ids:
            return
        old_columns = {class_id: index for index, class_id in enumerate(self.class_ids)}
        new_columns = {class_id: index for index, class_id in enumerate(updated)}
        Q = torch.zeros(
            (self.expand_dim, len(updated)), device=self.device, dtype=self.dtype
        )
        counts = torch.zeros(len(updated), device=self.device, dtype=self.dtype)
        for class_id, old in old_columns.items():
            new = new_columns[class_id]
            Q[:, new] = self.Q[:, old]
            counts[new] = self.counts[old]
        self.class_ids, self.Q, self.counts = updated, Q, counts

    def _targets(self, labels: torch.Tensor) -> torch.Tensor:
        columns = {class_id: index for index, class_id in enumerate(self.class_ids)}
        indices = torch.tensor(
            [columns[int(value)] for value in labels.detach().cpu().tolist()],
            device=self.device,
            dtype=torch.long,
        )
        return torch.nn.functional.one_hot(
            indices, num_classes=len(self.class_ids)
        ).to(self.dtype)

    def accumulate_codes(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        """Update sufficient statistics without retaining rows or solving yet."""
        Z = codes.to(device=self.device, dtype=self.dtype)
        y = labels.to(device=self.device, dtype=torch.long)
        if Z.ndim != 2 or Z.shape[1] != self.expand_dim:
            raise ValueError(f"codes must have shape (B, {self.expand_dim})")
        if y.ndim != 1 or len(y) != len(Z):
            raise ValueError("labels must have shape (B,)")
        if not bool(torch.isfinite(Z).all()):
            raise ValueError("codes contain NaN or Inf")
        if len(Z) == 0:
            return
        self._expand_classes(y)
        targets = self._targets(y)
        self.exact_diagonal += Z.square().sum(dim=0)
        self.Q += Z.T @ targets
        self.counts += targets.sum(dim=0)
        self.svd.update(Z)

    def finalize_update(self) -> None:
        """Rebuild the analytic classifier after one or more streamed batches."""
        if not self.class_ids:
            raise RuntimeError("cannot finalize an empty learner")
        self._recompute()
        self.assert_exemplar_free_state()

    def update_codes(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        self.accumulate_codes(codes, labels)
        self.finalize_update()

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.update_codes(self.encode(features), labels)

    def _recompute(self) -> None:
        tail = diagonal_tail(self.exact_diagonal, self.svd.U, self.svd.s)
        solution = solve_tail_ridge(
            self.svd.U,
            self.svd.s,
            tail,
            self.Q,
            self.ridge_lambda,
            solve_dtype=self.solver_dtype,
        )
        self.weights = solution.weights
        retained_diagonal = low_rank_diagonal(self.svd.U, self.svd.s)
        overshoot = (retained_diagonal - self.exact_diagonal).clamp_min(0)
        self.diagnostics.update(
            effective_rank=solution.active_rank,
            solver_relative_residual=solution.relative_residual,
            diagonal_tail_min=float(tail.min().item()),
            diagonal_tail_sum=float(tail.sum().item()),
            preclamp_diagonal_overshoot_max=float(overshoot.max().item()),
            total_rows=self.svd.total_rows,
        )

    def predict_logits_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("update() must be called before prediction")
        Z = codes.to(device=self.device, dtype=self.weights.dtype)
        if Z.ndim != 2 or Z.shape[1] != self.expand_dim:
            raise ValueError(f"codes must have shape (B, {self.expand_dim})")
        return Z @ self.weights

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.predict_logits_from_codes(self.encode(features))

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(features).argmax(1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[index] for index in columns])

    def persistent_tensors(self, *, include_classifier: bool = True) -> dict:
        tensors = {
            "projection": self.flyhash.projection_matrix,
            "U": self.svd.U,
            "s": self.svd.s,
            "exact_diagonal": self.exact_diagonal,
            "Q": self.Q,
            "counts": self.counts,
        }
        if include_classifier and self.weights is not None:
            tensors["weights"] = self.weights
        return tensors

    def persistent_state_bytes(self, *, include_classifier: bool = True) -> int:
        return sum(
            _storage_bytes(tensor)
            for tensor in self.persistent_tensors(
                include_classifier=include_classifier
            ).values()
        )

    def assert_exemplar_free_state(self) -> None:
        expected = {
            "projection": (self.expand_dim, self.feature_dim),
            "U": (self.expand_dim, self.svd.effective_rank),
            "s": (self.svd.effective_rank,),
            "exact_diagonal": (self.expand_dim,),
            "Q": (self.expand_dim, len(self.class_ids)),
            "counts": (len(self.class_ids),),
        }
        aggregate_tensors = self.persistent_tensors(include_classifier=False)
        for name, shape in expected.items():
            if tuple(aggregate_tensors[name].shape) != shape:
                raise AssertionError(f"invalid persistent tensor {name} shape")
        if self.weights is not None and self.weights.shape != self.Q.shape:
            raise AssertionError("invalid derived classifier shape")
        static_dimensions = {
            self.feature_dim,
            self.expand_dim,
            self.svd.effective_rank,
            len(self.class_ids),
        }
        if self.svd.total_rows not in static_dimensions:
            for name, tensor in self.persistent_tensors().items():
                if self.svd.total_rows in tensor.shape:
                    raise AssertionError(
                        f"persistent tensor {name} has historical sample dimension"
                    )

    def state_dict(self) -> dict:
        """Serialize fixed model and aggregates; derived weights are rebuilt."""
        return {
            "version": 2,
            "method": "tail_fly",
            "feature_dim": self.feature_dim,
            "expand_dim": self.expand_dim,
            "synaptic_degree": self.synaptic_degree,
            "coding_level": self.coding_level,
            "max_rank": self.max_rank,
            "ridge_lambda": self.ridge_lambda,
            "seed": self.seed,
            "solver_dtype": str(self.solver_dtype).removeprefix("torch."),
            "projection": self.flyhash.projection_matrix.detach().cpu(),
            "svd": self.svd.state_dict(),
            "exact_diagonal": self.exact_diagonal.detach().cpu().clone(),
            "Q": self.Q.detach().cpu().clone(),
            "counts": self.counts.detach().cpu().clone(),
            "class_ids": list(self.class_ids),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") not in {1, 2} or state.get("method") != "tail_fly":
            raise ValueError("unsupported TAIL-FLY checkpoint")
        for field in (
            "feature_dim",
            "expand_dim",
            "synaptic_degree",
            "coding_level",
            "max_rank",
            "ridge_lambda",
            "seed",
        ):
            if state.get(field) != getattr(self, field):
                raise ValueError(f"checkpoint configuration mismatch for {field}")
        recorded_solver_dtype = state.get(
            "solver_dtype", str(self.dtype).removeprefix("torch.")
        )
        if recorded_solver_dtype != str(self.solver_dtype).removeprefix("torch."):
            raise ValueError("checkpoint configuration mismatch for solver_dtype")
        projection = state["projection"].to(device=self.device)
        if projection.layout != torch.sparse_csc:
            raise ValueError("checkpoint projection must be sparse CSC")
        if projection.shape != (self.expand_dim, self.feature_dim):
            raise ValueError("checkpoint projection shape mismatch")
        if not bool(torch.isfinite(projection.values()).all()):
            raise ValueError("checkpoint projection contains NaN or Inf")
        self.flyhash.projection_matrix = projection
        self.svd.load_state_dict(state["svd"])
        self.exact_diagonal = state["exact_diagonal"].to(
            device=self.device, dtype=self.dtype
        )
        self.Q = state["Q"].to(device=self.device, dtype=self.dtype)
        self.counts = state["counts"].to(device=self.device, dtype=self.dtype)
        self.class_ids = [int(value) for value in state["class_ids"]]
        if self.class_ids != sorted(set(self.class_ids)):
            raise ValueError("checkpoint class IDs must be sorted and unique")
        if self.exact_diagonal.shape != (self.expand_dim,):
            raise ValueError("checkpoint diagonal shape mismatch")
        if self.Q.shape != (self.expand_dim, len(self.class_ids)):
            raise ValueError("checkpoint Q shape mismatch")
        if self.counts.shape != (len(self.class_ids),):
            raise ValueError("checkpoint counts shape mismatch")
        if not bool(
            torch.isfinite(self.exact_diagonal).all()
            and torch.isfinite(self.Q).all()
            and torch.isfinite(self.counts).all()
        ):
            raise ValueError("checkpoint statistics contain NaN or Inf")
        if bool((self.exact_diagonal < 0).any() or (self.counts < 0).any()):
            raise ValueError("checkpoint statistics must be non-negative")
        self._recompute()
        self.assert_exemplar_free_state()


def create_learner(**kwargs) -> TAILFlyLearner:
    return TAILFlyLearner(**kwargs)


assert "task_id" not in inspect.signature(TAILFlyLearner.predict_logits).parameters
