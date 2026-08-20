"""Exemplar-free CertiFLY learner with certified compressed Gram state."""

from __future__ import annotations

import inspect

import torch

from methods.flycl import select_ridge_parameter
from models.flyhash import FlyHash

from .quantization import QuantizedSymmetricGram
from .solver import CertifiedRidgeSolution, solve_certified_ridge


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


class CertiFLYLearner:
    """Fixed FLY/WTA representation with a quantized full-coordinate Gram."""

    is_exemplar_free = True

    def __init__(
        self,
        *,
        feature_dim: int,
        expand_dim: int,
        synaptic_degree: int,
        coding_level: float,
        block_size: int,
        error_fraction: float,
        max_bits: int,
        ridge_lower: float,
        ridge_upper: float,
        seed: int = 2025,
        device: str | torch.device = "cpu",
        statistics_dtype: torch.dtype = torch.float32,
        solver_dtype: torch.dtype = torch.float32,
        projection: torch.Tensor | None = None,
    ) -> None:
        if min(feature_dim, expand_dim, synaptic_degree, block_size) <= 0:
            raise ValueError("representation dimensions and block size must be positive")
        if synaptic_degree > feature_dim:
            raise ValueError("synaptic_degree cannot exceed feature_dim")
        if not 0 < coding_level <= 1:
            raise ValueError("coding_level must be in (0, 1]")
        if not 0 < error_fraction < 1:
            raise ValueError("error_fraction must be in (0, 1)")
        if max_bits not in {8, 16}:
            raise ValueError("max_bits must be 8 or 16")
        if ridge_lower >= ridge_upper:
            raise ValueError("invalid Ridge exponent range")
        if statistics_dtype not in {torch.float32, torch.float64}:
            raise ValueError("statistics_dtype must be float32 or float64")
        if solver_dtype not in {torch.float32, torch.float64}:
            raise ValueError("solver_dtype must be float32 or float64")
        self.feature_dim = int(feature_dim)
        self.expand_dim = int(expand_dim)
        self.synaptic_degree = int(synaptic_degree)
        self.coding_level = float(coding_level)
        self.block_size = int(block_size)
        self.error_fraction = float(error_fraction)
        self.max_bits = int(max_bits)
        self.ridge_lower = float(ridge_lower)
        self.ridge_upper = float(ridge_upper)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.statistics_dtype = statistics_dtype
        self.solver_dtype = solver_dtype

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

        self.gram: QuantizedSymmetricGram | None = None
        self.Q = torch.zeros(
            (self.expand_dim, 0), device=self.device, dtype=self.statistics_dtype
        )
        self.counts = torch.zeros((0,), device=self.device, dtype=self.statistics_dtype)
        self.class_ids: list[int] = []
        self.weights: torch.Tensor | None = None
        self.last_ridge: float | None = None
        self.total_rows = 0
        self.last_solution: CertifiedRidgeSolution | None = None
        self.diagnostics: dict = {
            "method": "certifly",
            "projection": "fixed_sparse_gaussian_wta",
            "gram_storage": "exact_diagonal_upper_correlation_blocks",
            "task_id_required": False,
            "error_fraction": self.error_fraction,
        }

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        values = features.to(
            device=self.device, dtype=self.flyhash.projection_matrix.dtype
        )
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B, {self.feature_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        return self.flyhash(values, self.coding_level, absolute_wta=False).to(
            self.statistics_dtype
        )

    def _expand_classes(self, labels: torch.Tensor) -> None:
        updated = sorted(set(self.class_ids) | set(map(int, labels.detach().cpu().tolist())))
        if updated == self.class_ids:
            return
        old_columns = {class_id: index for index, class_id in enumerate(self.class_ids)}
        new_columns = {class_id: index for index, class_id in enumerate(updated)}
        cross = torch.zeros(
            (self.expand_dim, len(updated)), device=self.device, dtype=self.statistics_dtype
        )
        counts = torch.zeros(len(updated), device=self.device, dtype=self.statistics_dtype)
        for class_id, old_column in old_columns.items():
            new_column = new_columns[class_id]
            cross[:, new_column] = self.Q[:, old_column]
            counts[new_column] = self.counts[old_column]
        self.class_ids, self.Q, self.counts = updated, cross, counts

    def _targets(self, labels: torch.Tensor) -> torch.Tensor:
        columns = {class_id: index for index, class_id in enumerate(self.class_ids)}
        indices = torch.tensor(
            [columns[int(value)] for value in labels.detach().cpu().tolist()],
            device=self.device,
            dtype=torch.long,
        )
        return torch.nn.functional.one_hot(indices, num_classes=len(self.class_ids)).to(
            self.statistics_dtype
        )

    def update_codes(
        self,
        codes: torch.Tensor,
        labels: torch.Tensor,
        *,
        selected_ridge: float | None = None,
    ) -> None:
        Z = codes.to(device=self.device, dtype=self.statistics_dtype)
        y = labels.to(device=self.device, dtype=torch.long)
        if Z.ndim != 2 or Z.shape[1] != self.expand_dim:
            raise ValueError(f"codes must have shape (B, {self.expand_dim})")
        if y.ndim != 1 or len(y) != len(Z) or not len(Z):
            raise ValueError("labels must be a non-empty vector aligned with codes")
        if not bool(torch.isfinite(Z).all()):
            raise ValueError("codes contain NaN or Inf")
        # Preserve the previous valid checkpoint until quantization and the
        # certified solve both succeed.  This matters when a new task exceeds
        # the configured bit-width/error budget.
        previous = {
            "gram": self.gram,
            "Q": self.Q,
            "counts": self.counts,
            "class_ids": self.class_ids,
            "weights": self.weights,
            "last_ridge": self.last_ridge,
            "total_rows": self.total_rows,
            "last_solution": self.last_solution,
            "diagnostics": dict(self.diagnostics),
        }
        try:
            self._expand_classes(y)
            targets = self._targets(y)
            ridge = (
                float(
                    select_ridge_parameter(
                        Z, targets, self.ridge_lower, self.ridge_upper
                    ).item()
                )
                if selected_ridge is None
                else float(selected_ridge)
            )
            if ridge <= 0:
                raise ValueError("selected Ridge value must be positive")

            delta_gram = Z.T @ Z
            if self.gram is None:
                self.gram = QuantizedSymmetricGram.from_dense(
                    delta_gram,
                    block_size=self.block_size,
                    ridge_lambda=ridge,
                    error_fraction=self.error_fraction,
                    max_bits=self.max_bits,
                )
            else:
                self.gram = self.gram.merge(
                    delta_gram,
                    ridge_lambda=ridge,
                    error_fraction=self.error_fraction,
                )
            self.Q = self.Q + Z.T @ targets
            self.counts = self.counts + targets.sum(dim=0)
            self.total_rows += len(Z)
            self.last_ridge = ridge
            self._recompute()
            self.assert_exemplar_free_state()
        except Exception:
            for name, value in previous.items():
                setattr(self, name, value)
            raise

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.update_codes(self.encode(features), labels)

    def _recompute(self) -> None:
        if self.gram is None or self.last_ridge is None:
            raise RuntimeError("cannot solve an empty learner")
        solution = solve_certified_ridge(
            self.gram,
            self.Q,
            self.last_ridge,
            solve_dtype=self.solver_dtype,
        )
        self.last_solution = solution
        self.weights = solution.weights
        histogram = self.gram.bit_histogram()
        self.diagnostics.update(
            selected_ridge=self.last_ridge,
            gram_error_bound=self.gram.error_bound,
            gram_error_fraction_of_ridge=self.gram.error_bound / self.last_ridge,
            last_quantization_error=self.gram.last_quantization_error,
            merge_count=self.gram.merge_count,
            int8_blocks=histogram[8],
            int16_blocks=histogram[16],
            solver_relative_residual=solution.relative_residual,
            relative_classifier_error_bound=solution.relative_classifier_error_bound,
            absolute_classifier_error_bound=solution.absolute_classifier_error_bound,
            total_rows=self.total_rows,
        )

    def predict_logits_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("update() must be called before prediction")
        values = codes.to(device=self.device, dtype=self.weights.dtype)
        if values.ndim != 2 or values.shape[1] != self.expand_dim:
            raise ValueError(f"codes must have shape (B, {self.expand_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("codes contain NaN or Inf")
        return values @ self.weights

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.predict_logits_from_codes(self.encode(features))

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(features).argmax(1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[index] for index in columns], dtype=torch.long)

    def persistent_tensors(self, *, include_classifier: bool = True) -> dict[str, torch.Tensor]:
        tensors = {
            "projection": self.flyhash.projection_matrix,
            "Q": self.Q,
            "counts": self.counts,
        }
        if self.gram is not None:
            tensors.update(self.gram.persistent_tensors())
        if include_classifier and self.weights is not None:
            tensors["weights"] = self.weights
        return tensors

    def persistent_state_bytes(self, *, include_classifier: bool = True) -> int:
        return sum(
            _storage_bytes(tensor)
            for tensor in self.persistent_tensors(include_classifier=include_classifier).values()
        )

    def persistent_state_summary(self) -> dict:
        entries = [
            {
                "name": name,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": _storage_bytes(tensor),
            }
            for name, tensor in self.persistent_tensors().items()
        ]
        return {"tensors": entries, "tensor_bytes": sum(item["bytes"] for item in entries)}

    def assert_exemplar_free_state(self) -> None:
        if self.Q.shape != (self.expand_dim, len(self.class_ids)):
            raise AssertionError("invalid Q shape")
        if self.counts.shape != (len(self.class_ids),):
            raise AssertionError("invalid counts shape")
        if self.weights is not None and self.weights.shape != self.Q.shape:
            raise AssertionError("invalid classifier shape")
        if self.gram is not None and self.gram.dimension != self.expand_dim:
            raise AssertionError("invalid Gram dimension")
        forbidden = ("history", "sample", "feature_cache", "labels", "codes")
        for name, tensor in self.persistent_tensors().items():
            if any(token in name.lower() for token in forbidden):
                raise AssertionError(f"forbidden sample-level persistent tensor: {name}")
            # Sample/features histories are matrices (or image tensors).  A
            # packed one-dimensional triangular block may coincidentally have
            # ``total_rows`` entries, so shape equality alone is not evidence
            # of replay for vectors.
            if tensor.ndim >= 2 and self.total_rows not in {
                self.feature_dim,
                self.expand_dim,
                len(self.class_ids),
            } and self.total_rows in tensor.shape:
                raise AssertionError(f"persistent tensor {name} has historical sample dimension")

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "method": "certifly",
            "feature_dim": self.feature_dim,
            "expand_dim": self.expand_dim,
            "synaptic_degree": self.synaptic_degree,
            "coding_level": self.coding_level,
            "block_size": self.block_size,
            "error_fraction": self.error_fraction,
            "max_bits": self.max_bits,
            "ridge_lower": self.ridge_lower,
            "ridge_upper": self.ridge_upper,
            "seed": self.seed,
            "statistics_dtype": str(self.statistics_dtype).removeprefix("torch."),
            "solver_dtype": str(self.solver_dtype).removeprefix("torch."),
            "projection": self.flyhash.projection_matrix.detach().cpu(),
            "gram": None if self.gram is None else self.gram.state_dict(),
            "Q": self.Q.detach().cpu().clone(),
            "counts": self.counts.detach().cpu().clone(),
            "class_ids": list(self.class_ids),
            "last_ridge": self.last_ridge,
            "total_rows": self.total_rows,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1 or state.get("method") != "certifly":
            raise ValueError("unsupported CertiFLY checkpoint")
        fields = (
            "feature_dim",
            "expand_dim",
            "synaptic_degree",
            "coding_level",
            "block_size",
            "error_fraction",
            "max_bits",
            "ridge_lower",
            "ridge_upper",
            "seed",
        )
        for field in fields:
            if state.get(field) != getattr(self, field):
                raise ValueError(f"checkpoint configuration mismatch for {field}")
        if state.get("statistics_dtype") != str(self.statistics_dtype).removeprefix("torch."):
            raise ValueError("checkpoint statistics dtype mismatch")
        if state.get("solver_dtype") != str(self.solver_dtype).removeprefix("torch."):
            raise ValueError("checkpoint solver dtype mismatch")
        projection = state["projection"].to(self.device)
        if projection.layout != torch.sparse_csc or projection.shape != (
            self.expand_dim,
            self.feature_dim,
        ):
            raise ValueError("invalid checkpoint projection")
        if not bool(torch.isfinite(projection.values()).all()):
            raise ValueError("checkpoint projection contains NaN or Inf")
        self.flyhash.projection_matrix = projection
        self.gram = (
            None
            if state["gram"] is None
            else QuantizedSymmetricGram.load_state_dict(state["gram"], device=self.device)
        )
        if self.gram is not None and (
            self.gram.block_size != self.block_size
            or self.gram.max_bits != self.max_bits
        ):
            raise ValueError("checkpoint quantized-Gram configuration mismatch")
        self.Q = state["Q"].to(device=self.device, dtype=self.statistics_dtype)
        self.counts = state["counts"].to(device=self.device, dtype=self.statistics_dtype)
        self.class_ids = [int(value) for value in state["class_ids"]]
        self.last_ridge = None if state["last_ridge"] is None else float(state["last_ridge"])
        self.total_rows = int(state["total_rows"])
        if self.class_ids != sorted(set(self.class_ids)):
            raise ValueError("checkpoint class IDs must be sorted and unique")
        if not bool(torch.isfinite(self.Q).all() and torch.isfinite(self.counts).all()):
            raise ValueError("checkpoint statistics contain NaN or Inf")
        if self.gram is None:
            if self.class_ids or self.last_ridge is not None or self.total_rows:
                raise ValueError("inconsistent empty CertiFLY checkpoint")
            self.weights = None
            self.last_solution = None
        else:
            if self.last_ridge is None or self.total_rows <= 0:
                raise ValueError("inconsistent populated CertiFLY checkpoint")
            self._recompute()
        self.assert_exemplar_free_state()


def create_learner(**kwargs) -> CertiFLYLearner:
    return CertiFLYLearner(**kwargs)


assert "task_id" not in inspect.signature(CertiFLYLearner.predict_logits).parameters
