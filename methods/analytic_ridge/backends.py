"""Generic additive-Ridge backends for fixed explicit feature streams."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .accounting import persistent_tensor_bytes
from .compressed_upper import CompressedUpper
from .qr import blocked_qr_rank_update


def _relative_residual(
    system: torch.Tensor, weights: torch.Tensor, cross: torch.Tensor
) -> float:
    numerator = torch.linalg.vector_norm(system @ weights - cross)
    denominator = max(float(torch.linalg.vector_norm(cross).item()), 1.0)
    return float(numerator.item()) / denominator


def _relative_factor_residual(
    factor: torch.Tensor, weights: torch.Tensor, cross: torch.Tensor
) -> float:
    numerator = torch.linalg.vector_norm(factor.T @ (factor @ weights) - cross)
    denominator = max(float(torch.linalg.vector_norm(cross).item()), 1.0)
    return float(numerator.item()) / denominator


class AnalyticRidgeBackend(ABC):
    """Base state for insertion-only class-incremental Ridge regression."""

    is_exemplar_free = True

    def __init__(
        self,
        *,
        dimension: int,
        ridge_lambda: float,
        device: str | torch.device = "cpu",
        statistics_dtype: torch.dtype = torch.float32,
        solver_dtype: torch.dtype = torch.float32,
    ) -> None:
        if dimension <= 0 or ridge_lambda <= 0:
            raise ValueError("dimension and ridge_lambda must be positive")
        if statistics_dtype not in {torch.float32, torch.float64}:
            raise ValueError("invalid statistics dtype")
        if solver_dtype not in {torch.float32, torch.float64}:
            raise ValueError("invalid solver dtype")
        self.dimension = int(dimension)
        self.ridge_lambda = float(ridge_lambda)
        self.device = torch.device(device)
        self.statistics_dtype = statistics_dtype
        self.solver_dtype = solver_dtype
        self.Q = torch.zeros(
            (self.dimension, 0), device=self.device, dtype=self.statistics_dtype
        )
        self.counts = torch.zeros(0, device=self.device, dtype=self.statistics_dtype)
        self.class_ids: list[int] = []
        self.weights: torch.Tensor | None = None
        self.total_rows = 0
        self.diagnostics: dict[str, object] = {
            "task_id_required": False,
            "ridge_lambda": self.ridge_lambda,
        }

    def _validated_batch(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = features.to(device=self.device, dtype=self.statistics_dtype)
        target_labels = labels.to(device=self.device, dtype=torch.long)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"features must have shape (B, {self.dimension})")
        if target_labels.ndim != 1 or len(target_labels) != len(values) or not len(values):
            raise ValueError("labels must align with a non-empty feature matrix")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        return values, target_labels

    def _expanded_statistics(
        self, labels: torch.Tensor
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        updated = sorted(set(self.class_ids) | set(map(int, labels.cpu().tolist())))
        old_columns = {value: index for index, value in enumerate(self.class_ids)}
        new_columns = {value: index for index, value in enumerate(updated)}
        cross = torch.zeros(
            (self.dimension, len(updated)),
            device=self.device,
            dtype=self.statistics_dtype,
        )
        counts = torch.zeros(
            len(updated), device=self.device, dtype=self.statistics_dtype
        )
        for class_id, old_column in old_columns.items():
            column = new_columns[class_id]
            cross[:, column] = self.Q[:, old_column]
            counts[column] = self.counts[old_column]
        columns = torch.tensor(
            [new_columns[int(value)] for value in labels.cpu().tolist()],
            device=self.device,
            dtype=torch.long,
        )
        targets = torch.nn.functional.one_hot(
            columns, num_classes=len(updated)
        ).to(self.statistics_dtype)
        return updated, cross, counts, targets

    @abstractmethod
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        raise NotImplementedError

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("backend has not been updated")
        values = features.to(device=self.device, dtype=self.weights.dtype)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"features must have shape (B, {self.dimension})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        return values @ self.weights

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(features).argmax(1).cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns])

    def _common_persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {"Q": self.Q, "counts": self.counts}
        if self.weights is not None:
            tensors["weights"] = self.weights
        return tensors

    @abstractmethod
    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def persistent_state_bytes(self) -> int:
        return persistent_tensor_bytes(self.persistent_tensors())

    def assert_exemplar_free_state(self) -> None:
        if self.Q.shape != (self.dimension, len(self.class_ids)):
            raise AssertionError("invalid Q shape")
        if self.counts.shape != (len(self.class_ids),):
            raise AssertionError("invalid class-count shape")
        if self.weights is not None and self.weights.shape != self.Q.shape:
            raise AssertionError("invalid classifier shape")
        forbidden = ("history", "sample", "feature_cache", "labels", "codes")
        for name, tensor in self.persistent_tensors().items():
            if any(token in name.lower() for token in forbidden):
                raise AssertionError(f"forbidden sample-level state: {name}")
            if (
                tensor.ndim >= 2
                and self.total_rows not in {
                    self.dimension,
                    len(self.class_ids),
                }
                and self.total_rows in tensor.shape
            ):
                raise AssertionError(f"historical sample dimension in {name}")

    def _common_state(self) -> dict[str, object]:
        return {
            "version": 1,
            "dimension": self.dimension,
            "ridge_lambda": self.ridge_lambda,
            "statistics_dtype": str(self.statistics_dtype).removeprefix("torch."),
            "solver_dtype": str(self.solver_dtype).removeprefix("torch."),
            "Q": self.Q.detach().cpu().clone(),
            "counts": self.counts.detach().cpu().clone(),
            "class_ids": list(self.class_ids),
            "total_rows": self.total_rows,
        }

    def _load_common_values(self, state: dict, *, dimension_field: str) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported analytic backend checkpoint")
        if int(state[dimension_field]) != self.dimension:
            raise ValueError("checkpoint dimension mismatch")
        if float(state["ridge_lambda"]) != self.ridge_lambda:
            raise ValueError("checkpoint ridge_lambda mismatch")
        if state.get("statistics_dtype") != str(self.statistics_dtype).removeprefix(
            "torch."
        ):
            raise ValueError("checkpoint statistics dtype mismatch")
        if state.get("solver_dtype") != str(self.solver_dtype).removeprefix("torch."):
            raise ValueError("checkpoint solver dtype mismatch")
        self.Q = state["Q"].to(device=self.device, dtype=self.statistics_dtype)
        self.counts = state["counts"].to(
            device=self.device, dtype=self.statistics_dtype
        )
        self.class_ids = [int(value) for value in state["class_ids"]]
        self.total_rows = int(state["total_rows"])
        if self.class_ids != sorted(set(self.class_ids)):
            raise ValueError("checkpoint class IDs must be sorted and unique")
        if self.Q.shape != (self.dimension, len(self.class_ids)):
            raise ValueError("invalid checkpoint Q shape")
        if self.counts.shape != (len(self.class_ids),):
            raise ValueError("invalid checkpoint count shape")
        if not bool(torch.isfinite(self.Q).all()) or not bool(
            torch.isfinite(self.counts).all()
        ):
            raise ValueError("checkpoint statistics contain NaN or Inf")
        if bool((self.counts < 0).any()):
            raise ValueError("checkpoint counts must be nonnegative")
        if abs(float(self.counts.sum()) - self.total_rows) > 1e-3:
            raise ValueError("checkpoint counts do not match total_rows")


class ExactGramBackend(AnalyticRidgeBackend):
    """Dense FP statistics reference for the generic backend contract."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gram = torch.zeros(
            (self.dimension, self.dimension),
            device=self.device,
            dtype=self.statistics_dtype,
        )
        self.diagnostics["method"] = "exact_gram"

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        values, target_labels = self._validated_batch(features, labels)
        class_ids, cross, counts, targets = self._expanded_statistics(target_labels)
        self.gram.add_(values.T @ values)
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        system = self.gram.to(self.solver_dtype).clone()
        system.diagonal().add_(self.ridge_lambda)
        symmetric = (system + system.T) * 0.5
        factor, info = torch.linalg.cholesky_ex(symmetric)
        if int(info.max().item()) != 0:
            raise RuntimeError("Exact Gram Ridge system is not positive definite")
        work_cross = new_cross.to(self.solver_dtype)
        weights = torch.cholesky_solve(work_cross, factor)
        residual = _relative_residual(symmetric, weights, work_cross)
        self.class_ids, self.Q, self.counts = class_ids, new_cross, new_counts
        self.weights = weights
        self.total_rows += len(values)
        self.diagnostics.update(
            solver_relative_residual=residual, total_rows=self.total_rows
        )
        self.assert_exemplar_free_state()

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = self._common_persistent_tensors()
        tensors["gram"] = self.gram
        return tensors

    def state_dict(self) -> dict[str, object]:
        state = self._common_state()
        state.update(
            method="exact_gram_analytic_ridge",
            gram=self.gram.detach().cpu().clone(),
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        if state.get("method") != "exact_gram_analytic_ridge":
            raise ValueError("invalid Exact Gram analytic checkpoint")
        self._load_common_values(state, dimension_field="dimension")
        self.gram = state["gram"].to(
            device=self.device, dtype=self.statistics_dtype
        )
        if self.gram.shape != (self.dimension, self.dimension):
            raise ValueError("invalid checkpoint Gram shape")
        if not bool(torch.isfinite(self.gram).all()):
            raise ValueError("checkpoint Gram contains NaN or Inf")
        if not torch.equal(self.gram, self.gram.T):
            raise ValueError("checkpoint Gram must be exactly symmetric")
        if self.total_rows:
            system = self.gram.to(self.solver_dtype).clone()
            system.diagonal().add_(self.ridge_lambda)
            factor, info = torch.linalg.cholesky_ex(system)
            if int(info.max().item()) != 0:
                raise ValueError("checkpoint Gram Ridge system is not positive definite")
            work_cross = self.Q.to(self.solver_dtype)
            self.weights = torch.cholesky_solve(work_cross, factor)
            self.diagnostics["solver_relative_residual"] = _relative_residual(
                system, self.weights, work_cross
            )
        else:
            self.weights = None
        self.assert_exemplar_free_state()


class SquareRootBackend(AnalyticRidgeBackend):
    """Full-rank mixed-precision square-root backend for additive Ridge."""

    def __init__(
        self,
        *,
        storage_mode: str,
        block_size: int,
        group_size: int,
        update_panel_size: int = 128,
        update_trailing_chunk_size: int | None = None,
        first_update_backend: str = "gram_cholesky",
        quantization_backend: str = "streaming",
        quantization_batch_blocks: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if storage_mode not in {"float16", "int8"}:
            raise ValueError("storage_mode must be float16 or int8")
        if min(block_size, group_size, update_panel_size, quantization_batch_blocks) <= 0:
            raise ValueError("storage and update sizes must be positive")
        if update_trailing_chunk_size is not None and update_trailing_chunk_size <= 0:
            raise ValueError("trailing chunk size must be positive")
        if first_update_backend not in {"gram_cholesky", "implicit_ridge_qr"}:
            raise ValueError("invalid first update backend")
        if quantization_backend not in {"eager", "streaming"}:
            raise ValueError("invalid quantization backend")
        self.storage_mode = storage_mode
        self.block_size = int(block_size)
        self.group_size = int(group_size)
        self.update_panel_size = int(update_panel_size)
        self.update_trailing_chunk_size = update_trailing_chunk_size
        self.first_update_backend = first_update_backend
        self.quantization_backend = quantization_backend
        self.quantization_batch_blocks = int(quantization_batch_blocks)
        self.factor: CompressedUpper | None = None
        self.diagnostics.update(
            method="square_root",
            storage=storage_mode,
            structurally_spd=True,
            update_backend="blocked_qr",
        )

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        values, target_labels = self._validated_batch(features, labels)
        class_ids, cross, counts, targets = self._expanded_statistics(target_labels)
        solve_values = values.to(self.solver_dtype)
        previous = (
            None
            if self.factor is None
            else self.factor.reconstruct_upper(dtype=self.solver_dtype)
        )

        if previous is None and self.first_update_backend == "implicit_ridge_qr":
            previous = torch.zeros(
                (self.dimension, self.dimension),
                device=self.device,
                dtype=self.solver_dtype,
            )
            previous.diagonal().fill_(self.ridge_lambda**0.5)

        if previous is not None:
            exact_upper = blocked_qr_rank_update(
                previous,
                solve_values,
                panel_size=self.update_panel_size,
                trailing_chunk_size=self.update_trailing_chunk_size,
            )
        else:
            system = solve_values.T @ solve_values
            system.diagonal().add_(self.ridge_lambda)
            symmetric = (system + system.T) * 0.5
            lower, info = torch.linalg.cholesky_ex(symmetric)
            if int(info.max().item()) != 0:
                raise RuntimeError("square-root first update failed Cholesky")
            exact_upper = lower.T

        if self.quantization_backend == "streaming":
            compressed, relative_factor_error = (
                CompressedUpper.from_upper_inplace_streaming(
                    exact_upper,
                    block_size=self.block_size,
                    group_size=self.group_size,
                    mode=self.storage_mode,
                    maximum_batched_blocks=self.quantization_batch_blocks,
                )
            )
            reconstructed = exact_upper
        else:
            compressed = CompressedUpper.from_upper(
                exact_upper,
                block_size=self.block_size,
                group_size=self.group_size,
                mode=self.storage_mode,
            )
            reconstructed = compressed.reconstruct_upper(dtype=self.solver_dtype)
            relative_factor_error = float(
                torch.dist(reconstructed, exact_upper).item()
            ) / max(float(torch.linalg.vector_norm(exact_upper).item()), 1.0)

        if bool((reconstructed.diagonal() <= 0).any()):
            raise RuntimeError("compressed square-root diagonal is not positive")
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        work_cross = new_cross.to(self.solver_dtype)
        intermediate = torch.linalg.solve_triangular(
            reconstructed.T, work_cross, upper=False
        )
        weights = torch.linalg.solve_triangular(
            reconstructed, intermediate, upper=True
        )
        residual = _relative_factor_residual(reconstructed, weights, work_cross)

        self.factor = compressed
        self.class_ids, self.Q, self.counts = class_ids, new_cross, new_counts
        self.weights = weights
        self.total_rows += len(values)
        self.diagnostics.update(
            solver_relative_residual=residual,
            relative_local_factor_error=relative_factor_error,
            total_rows=self.total_rows,
        )
        self.assert_exemplar_free_state()

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = self._common_persistent_tensors()
        if self.factor is not None:
            tensors.update(self.factor.persistent_tensors("factor"))
        return tensors

    def state_dict(self) -> dict[str, object]:
        state = self._common_state()
        state.update(
            method="square_root_analytic_ridge",
            storage_mode=self.storage_mode,
            block_size=self.block_size,
            group_size=self.group_size,
            update_panel_size=self.update_panel_size,
            update_trailing_chunk_size=self.update_trailing_chunk_size,
            first_update_backend=self.first_update_backend,
            quantization_backend=self.quantization_backend,
            quantization_batch_blocks=self.quantization_batch_blocks,
            factor=None if self.factor is None else self.factor.state_dict(),
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        if state.get("method") != "square_root_analytic_ridge":
            raise ValueError("invalid square-root analytic checkpoint")
        self._load_configured_state(
            state, dimension_field="dimension", legacy=False
        )

    def load_legacy_srq_fly_state_dict(self, state: dict) -> None:
        if state.get("method") != "square_root_fly":
            raise ValueError("invalid legacy SRQ-FLY checkpoint")
        if state.get("update_backend") != "blocked_qr":
            raise ValueError("legacy checkpoint does not use blocked_qr")
        self._load_configured_state(
            state, dimension_field="expand_dim", legacy=True
        )

    def _load_configured_state(
        self, state: dict, *, dimension_field: str, legacy: bool
    ) -> None:
        expected = {
            "storage_mode": self.storage_mode,
            "block_size": self.block_size,
            "group_size": self.group_size,
            "update_panel_size": self.update_panel_size,
            "update_trailing_chunk_size": self.update_trailing_chunk_size,
            "first_update_backend": self.first_update_backend,
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(f"checkpoint configuration mismatch for {field}")
        if not legacy:
            if state.get("quantization_backend") != self.quantization_backend:
                raise ValueError("checkpoint quantization backend mismatch")
            if (
                state.get("quantization_batch_blocks")
                != self.quantization_batch_blocks
            ):
                raise ValueError("checkpoint quantization batch size mismatch")
        self._load_common_values(state, dimension_field=dimension_field)
        self.factor = (
            None
            if state["factor"] is None
            else CompressedUpper.load_state_dict(state["factor"], device=self.device)
        )
        if self.factor is None:
            if self.total_rows:
                raise ValueError("non-empty checkpoint is missing factor state")
            self.weights = None
        else:
            if (
                self.factor.dimension != self.dimension
                or self.factor.block_size != self.block_size
                or self.factor.group_size != self.group_size
                or self.factor.mode != self.storage_mode
            ):
                raise ValueError("compressed factor configuration mismatch")
            reconstructed = self.factor.reconstruct_upper(dtype=self.solver_dtype)
            work_cross = self.Q.to(self.solver_dtype)
            intermediate = torch.linalg.solve_triangular(
                reconstructed.T, work_cross, upper=False
            )
            self.weights = torch.linalg.solve_triangular(
                reconstructed, intermediate, upper=True
            )
            self.diagnostics["solver_relative_residual"] = _relative_factor_residual(
                reconstructed, self.weights, work_cross
            )
        self.assert_exemplar_free_state()
