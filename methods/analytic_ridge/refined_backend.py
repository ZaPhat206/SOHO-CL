"""Scale-refined, same-byte INT8 square-root Ridge backend."""

from __future__ import annotations

import torch

from methods.srq_fly_optimized.storage import CompressedUpper

from .backends import AnalyticRidgeBackend, _relative_factor_residual
from .qr import blocked_qr_rank_update
from .refined_upper import compress_upper_refined_inplace_streaming


class ScaleRefinedSquareRootBackend(AnalyticRidgeBackend):
    """P2B-compatible INT8 payload with deterministic LS-refined scales."""

    def __init__(
        self,
        *,
        block_size: int,
        group_size: int,
        update_panel_size: int = 128,
        update_trailing_chunk_size: int | None = None,
        first_update_backend: str = "gram_cholesky",
        quantization_batch_blocks: int = 64,
        scale_refinement_iterations: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if min(
            block_size,
            group_size,
            update_panel_size,
            quantization_batch_blocks,
            scale_refinement_iterations,
        ) <= 0:
            raise ValueError("storage, update, and refinement sizes must be positive")
        if update_trailing_chunk_size is not None and update_trailing_chunk_size <= 0:
            raise ValueError("trailing chunk size must be positive")
        if first_update_backend != "gram_cholesky":
            raise ValueError("M11b supports only the locked Gram-Cholesky initializer")
        self.block_size = int(block_size)
        self.group_size = int(group_size)
        self.update_panel_size = int(update_panel_size)
        self.update_trailing_chunk_size = update_trailing_chunk_size
        self.first_update_backend = first_update_backend
        self.quantization_batch_blocks = int(quantization_batch_blocks)
        self.scale_refinement_iterations = int(scale_refinement_iterations)
        self.factor: CompressedUpper | None = None
        self.diagnostics.update(
            method="scale_refined_square_root",
            storage="int8",
            scale_rule="alternating_least_squares",
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
                raise RuntimeError("scale-refined first update failed Cholesky")
            exact_upper = lower.T

        compressed, relative_error, maxabs_error = (
            compress_upper_refined_inplace_streaming(
                exact_upper,
                block_size=self.block_size,
                group_size=self.group_size,
                maximum_batched_blocks=self.quantization_batch_blocks,
                refinement_iterations=self.scale_refinement_iterations,
            )
        )
        reconstructed = exact_upper
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
            relative_local_factor_error=relative_error,
            maxabs_same_input_relative_factor_error=maxabs_error,
            scale_refinement_iterations=self.scale_refinement_iterations,
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
            method="scale_refined_square_root_analytic_ridge",
            block_size=self.block_size,
            group_size=self.group_size,
            update_panel_size=self.update_panel_size,
            update_trailing_chunk_size=self.update_trailing_chunk_size,
            first_update_backend=self.first_update_backend,
            quantization_batch_blocks=self.quantization_batch_blocks,
            scale_refinement_iterations=self.scale_refinement_iterations,
            factor=None if self.factor is None else self.factor.state_dict(),
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        if state.get("method") != "scale_refined_square_root_analytic_ridge":
            raise ValueError("invalid scale-refined checkpoint")
        expected = {
            "block_size": self.block_size,
            "group_size": self.group_size,
            "update_panel_size": self.update_panel_size,
            "update_trailing_chunk_size": self.update_trailing_chunk_size,
            "first_update_backend": self.first_update_backend,
            "quantization_batch_blocks": self.quantization_batch_blocks,
            "scale_refinement_iterations": self.scale_refinement_iterations,
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(f"checkpoint configuration mismatch for {field}")
        self._load_common_values(state, dimension_field="dimension")
        if state["factor"] is None:
            if self.total_rows:
                raise ValueError("non-empty checkpoint is missing factor state")
            self.factor = None
            self.weights = None
        else:
            self.factor = CompressedUpper.load_state_dict(
                state["factor"], device=self.device
            )
            if (
                self.factor.dimension != self.dimension
                or self.factor.block_size != self.block_size
                or self.factor.group_size != self.group_size
                or self.factor.mode != "int8"
            ):
                raise ValueError("compressed factor configuration mismatch")
            reconstructed = self.factor.reconstruct_upper(dtype=self.solver_dtype)
            if bool((reconstructed.diagonal() <= 0).any()):
                raise ValueError("checkpoint factor diagonal is not positive")
            work_cross = self.Q.to(self.solver_dtype)
            intermediate = torch.linalg.solve_triangular(
                reconstructed.T, work_cross, upper=False
            )
            self.weights = torch.linalg.solve_triangular(
                reconstructed, intermediate, upper=True
            )
            self.diagnostics["solver_relative_residual"] = (
                _relative_factor_residual(reconstructed, self.weights, work_cross)
            )
        self.assert_exemplar_free_state()


__all__ = ["ScaleRefinedSquareRootBackend"]
