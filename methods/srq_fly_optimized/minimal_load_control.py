"""Uncertified minimal-load direct-Gram control for M21.

Priority 3 repaired the direct-INT8 Gram matrix with the smallest diagonal load
that Weyl's inequality certifies.  A reviewer may ask whether a much smaller,
uncertified load would suffice.  This control adds, at every task, the
smallest load on a fixed geometric grid for which the FP32 Cholesky
factorization succeeds.  It is kept outside the locked learner and certified
control modules so their source identities stay reproducible.
"""

from __future__ import annotations

import torch

from .direct_control import _quantization_error_metrics
from .learner import _BaseFLYLearner, _cholesky_solve
from .storage import CompressedUpper


def _cholesky_succeeds(system: torch.Tensor) -> bool:
    # Same symmetrization and factorization as ``_cholesky_solve``.
    _, info = torch.linalg.cholesky_ex((system + system.T) * 0.5)
    return int(info.max().item()) == 0


def _set_loaded_diagonal(
    matrix: torch.Tensor, original_diagonal: torch.Tensor, total_load: float
) -> None:
    matrix.diagonal().copy_(original_diagonal + total_load)


def grid_load(base_load: float, index: int, steps_per_doubling: int) -> float:
    return float(base_load) * 2.0 ** (index / steps_per_doubling)


def minimal_cholesky_load(
    matrix: torch.Tensor,
    *,
    ridge_lambda: float,
    base_load: float,
    steps_per_doubling: int,
    maximum_load: float,
) -> dict:
    """Smallest grid load for which ``matrix + (lambda + load) I`` factors.

    The candidates are 0 and ``base_load * 2**(j / steps_per_doubling)`` for
    ``j = 0, 1, ...``.  The search doubles the load until the FP32 Cholesky
    factorization succeeds and then bisects the grid index between the last
    failure and the first success, so it assumes success is monotone in the
    load.  ``matrix`` is restored to its input values before returning.
    """
    if base_load <= 0 or maximum_load <= base_load or steps_per_doubling <= 0:
        raise ValueError("invalid minimal-load grid")
    original = matrix.diagonal().clone()
    attempts = 0

    def succeeds(load: float) -> bool:
        nonlocal attempts
        attempts += 1
        _set_loaded_diagonal(matrix, original, ridge_lambda + load)
        return _cholesky_succeeds(matrix)

    try:
        if succeeds(0.0):
            return {"load": 0.0, "grid_index": None, "attempts": attempts,
                    "zero_load_succeeded": True}
        failed, index = -1, 0
        while not succeeds(grid_load(base_load, index, steps_per_doubling)):
            failed = index
            index += steps_per_doubling
            if grid_load(base_load, index, steps_per_doubling) > maximum_load:
                raise RuntimeError("minimal-load search exceeded its maximum load")
        succeeded = index
        while succeeded - failed > 1:
            middle = (failed + succeeded) // 2
            if succeeds(grid_load(base_load, middle, steps_per_doubling)):
                succeeded = middle
            else:
                failed = middle
        return {
            "load": grid_load(base_load, succeeded, steps_per_doubling),
            "grid_index": succeeded,
            "attempts": attempts,
            "zero_load_succeeded": False,
        }
    finally:
        matrix.diagonal().copy_(original)


class MinimalLoadDirectInt8GramLearner(_BaseFLYLearner):
    """Direct-INT8 Gram with the smallest grid load that lets Cholesky succeed.

    The stored Gram matrix, its quantization and the solve are those of the
    certified control; only the diagonal load differs.  The load is chosen
    from the quantized matrix alone and uses no labels or accuracy.
    """

    def __init__(
        self,
        *,
        load_margin_multiplier: float = 8.0,
        load_grid_steps_per_doubling: int = 4,
        maximum_load_to_ridge_ratio: float = 1.0e6,
        error_chunk_size: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if load_margin_multiplier <= 0 or load_grid_steps_per_doubling <= 0:
            raise ValueError("invalid minimal-load grid")
        if maximum_load_to_ridge_ratio <= 0 or error_chunk_size <= 0:
            raise ValueError("invalid minimal-load limits")
        self.load_margin_multiplier = float(load_margin_multiplier)
        self.load_grid_steps_per_doubling = int(load_grid_steps_per_doubling)
        self.maximum_load_to_ridge_ratio = float(maximum_load_to_ridge_ratio)
        self.error_chunk_size = int(error_chunk_size)
        self.gram: CompressedUpper | None = None
        self.diagonal_loading = torch.zeros((), device=self.device, dtype=torch.float64)
        self.diagnostics.update(
            method="minimal_load_direct_int8_gram",
            storage="groupwise_int8",
            repair="smallest_grid_load_with_successful_fp32_cholesky",
        )

    def _grid_base_load(self, reconstructed: torch.Tensor) -> float:
        # The fixed floating-point margin of the certified control.
        scale = max(
            float(reconstructed.diagonal().abs().amax().item()), self.ridge_lambda, 1.0
        )
        return (
            self.load_margin_multiplier
            * torch.finfo(self.solver_dtype).eps
            * self.expand_dim
            * scale
        )

    def update_codes(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        values = codes.to(device=self.device, dtype=self.statistics_dtype)
        target_labels = labels.to(device=self.device, dtype=torch.long)
        if values.ndim != 2 or values.shape[1] != self.expand_dim:
            raise ValueError(f"codes must have shape (B, {self.expand_dim})")
        if target_labels.ndim != 1 or len(target_labels) != len(values) or not len(values):
            raise ValueError("labels must align with a non-empty code matrix")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("codes contain NaN or Inf")

        class_ids, cross, counts, targets = self._expanded_statistics(target_labels)
        updated = values.T @ values
        if self.gram is not None:
            updated.add_(self.gram.reconstruct_symmetric(dtype=self.statistics_dtype))
        updated = (updated + updated.T) * 0.5
        compressed = CompressedUpper.from_upper(
            updated, block_size=self.block_size, group_size=self.group_size, mode="int8"
        )
        reconstructed = compressed.reconstruct_symmetric(dtype=self.solver_dtype)
        infinity_error, relative_storage_error = _quantization_error_metrics(
            reconstructed.to(updated.dtype), updated, row_chunk_size=self.error_chunk_size
        )
        del updated

        base_load = self._grid_base_load(reconstructed)
        search = minimal_cholesky_load(
            reconstructed,
            ridge_lambda=self.ridge_lambda,
            base_load=base_load,
            steps_per_doubling=self.load_grid_steps_per_doubling,
            maximum_load=self.maximum_load_to_ridge_ratio * self.ridge_lambda,
        )
        loading = float(search["load"])

        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        work_cross = new_cross.to(self.solver_dtype)
        original = reconstructed.diagonal().clone()
        _set_loaded_diagonal(reconstructed, original, self.ridge_lambda + loading)
        weights, residual = _cholesky_solve(reconstructed, work_cross)

        self.gram = compressed
        self.class_ids, self.Q, self.counts = class_ids, new_cross, new_counts
        self.weights = weights
        self.total_rows += len(values)
        self.diagonal_loading.fill_(loading)
        self.diagnostics.update(
            solver_relative_residual=residual,
            relative_local_storage_error=relative_storage_error,
            local_quantization_error_infinity_bound=infinity_error,
            grid_base_load=base_load,
            grid_index=search["grid_index"],
            cholesky_attempts=search["attempts"],
            zero_load_succeeded=search["zero_load_succeeded"],
            diagonal_loading=loading,
            effective_ridge_lambda=self.ridge_lambda + loading,
            total_rows=self.total_rows,
        )
        self.assert_exemplar_free_state()

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = self._base_persistent_tensors()
        tensors["diagonal_loading"] = self.diagonal_loading
        if self.gram is not None:
            tensors.update(self.gram.persistent_tensors("gram"))
        return tensors

    def state_dict(self) -> dict:
        state = self._configuration_state()
        state.update(
            method="minimal_load_direct_int8_gram",
            load_margin_multiplier=self.load_margin_multiplier,
            load_grid_steps_per_doubling=self.load_grid_steps_per_doubling,
            maximum_load_to_ridge_ratio=self.maximum_load_to_ridge_ratio,
            error_chunk_size=self.error_chunk_size,
            diagonal_loading=self.diagonal_loading.detach().cpu().clone(),
            gram=None if self.gram is None else self.gram.state_dict(),
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        self._load_common(state, "minimal_load_direct_int8_gram")
        for field in (
            "load_margin_multiplier", "load_grid_steps_per_doubling",
            "maximum_load_to_ridge_ratio", "error_chunk_size",
        ):
            if state.get(field) != getattr(self, field):
                raise ValueError(f"checkpoint configuration mismatch for {field}")
        loading = state["diagonal_loading"].to(device=self.device, dtype=torch.float64)
        if loading.shape or not bool(torch.isfinite(loading)) or float(loading) < 0:
            raise ValueError("invalid minimal-load checkpoint loading")
        self.diagonal_loading.copy_(loading)
        self.gram = None if state["gram"] is None else CompressedUpper.load_state_dict(
            state["gram"], device=self.device
        )
        if self.gram is None:
            if self.total_rows or self.class_ids:
                raise ValueError("non-empty checkpoint is missing its Gram state")
            self.weights = None
        else:
            self._validate_compressed(self.gram, mode="int8")
            reconstructed = self.gram.reconstruct_symmetric(dtype=self.solver_dtype)
            original = reconstructed.diagonal().clone()
            _set_loaded_diagonal(
                reconstructed, original, self.ridge_lambda + float(loading.item())
            )
            self.weights, residual = _cholesky_solve(
                reconstructed, self.Q.to(self.solver_dtype)
            )
            self.diagnostics.update(
                solver_relative_residual=residual,
                diagonal_loading=float(loading.item()),
                effective_ridge_lambda=self.ridge_lambda + float(loading.item()),
            )
        self.assert_exemplar_free_state()
