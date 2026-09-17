"""Storage-efficient exact and low-rank controls for analytic Ridge.

These backends are deliberately separate from the production SRQ backends.
They support the equal-persistent-memory experiment in which an FP32 symmetric
Gram matrix is stored without duplicate entries, or replaced by a deterministic
Frequent Directions sketch.  Neither backend retains sample-level state.
"""

from __future__ import annotations

import math

import torch

from .accounting import persistent_tensor_bytes
from .backends import AnalyticRidgeBackend


def packed_upper_element_count(dimension: int) -> int:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return dimension * (dimension + 1) // 2


def packed_exact_backend_bytes(
    dimension: int,
    classes: int,
    *,
    element_size: int = 4,
) -> int:
    """Final backend bytes: packed Gram, Q, weights, and class counts."""

    if dimension <= 0 or classes <= 0 or element_size <= 0:
        raise ValueError("invalid packed Exact accounting arguments")
    return (
        element_size * packed_upper_element_count(dimension)
        + 2 * element_size * dimension * classes
        + element_size * classes
    )


def frequent_directions_backend_bytes(
    dimension: int,
    classes: int,
    rank: int,
    *,
    statistics_element_size: int = 4,
    correction_element_size: int = 8,
) -> int:
    """Final compact FD bytes: sketch, Q, counts, and correction.

    The classifier is held as the compact Woodbury coefficient ``C`` in
    ``W = Q/lambda - B.T C`` rather than as a separately stored dense FP32
    matrix.  ``C`` is FP64 and its bytes are part of the persistent budget.
    """

    if dimension <= 0 or classes <= 0 or rank <= 0 or rank > dimension:
        raise ValueError("invalid Frequent Directions accounting arguments")
    if statistics_element_size <= 0 or correction_element_size <= 0:
        raise ValueError("element sizes must be positive")
    return (
        statistics_element_size * rank * dimension
        + statistics_element_size * dimension * classes
        + statistics_element_size * classes
        + correction_element_size * rank * classes
    )


def largest_packed_exact_dimension(
    *,
    target_total_bytes: int,
    feature_dimension: int,
    classes: int,
    maximum_dimension: int,
    element_size: int = 4,
) -> tuple[int, int]:
    """Largest expanded width whose projection and packed backend fit."""

    if min(target_total_bytes, feature_dimension, classes, maximum_dimension) <= 0:
        raise ValueError("invalid packed Exact budget arguments")
    best_dimension = 0
    best_bytes = 0
    for dimension in range(1, maximum_dimension + 1):
        total = (
            element_size * feature_dimension * dimension
            + packed_exact_backend_bytes(
                dimension, classes, element_size=element_size
            )
        )
        if total <= target_total_bytes:
            best_dimension, best_bytes = dimension, total
        else:
            break
    if not best_dimension:
        raise ValueError("target budget cannot fit packed Exact at dimension one")
    return best_dimension, best_bytes


def largest_frequent_directions_rank(
    *,
    target_total_bytes: int,
    feature_dimension: int,
    expanded_dimension: int,
    classes: int,
    statistics_element_size: int = 4,
    correction_element_size: int = 8,
) -> tuple[int, int]:
    """Largest FD rank whose projection and final backend fit."""

    if min(
        target_total_bytes, feature_dimension, expanded_dimension, classes
    ) <= 0:
        raise ValueError("invalid Frequent Directions budget arguments")
    projection_bytes = statistics_element_size * feature_dimension * expanded_dimension
    base = projection_bytes + statistics_element_size * expanded_dimension * classes
    base += statistics_element_size * classes
    per_rank = statistics_element_size * expanded_dimension + correction_element_size * classes
    rank = min(expanded_dimension, (target_total_bytes - base) // per_rank)
    if rank <= 0:
        raise ValueError("target budget cannot fit a rank-one FD sketch")
    total = projection_bytes + frequent_directions_backend_bytes(
        expanded_dimension,
        classes,
        int(rank),
        statistics_element_size=statistics_element_size,
        correction_element_size=correction_element_size,
    )
    return int(rank), int(total)


def _relative_implicit_residual(
    sketch: torch.Tensor,
    ridge_lambda: float,
    weights: torch.Tensor,
    cross: torch.Tensor,
) -> float:
    residual = sketch.T @ (sketch @ weights)
    residual.add_(weights, alpha=float(ridge_lambda)).sub_(cross)
    denominator = max(float(torch.linalg.vector_norm(cross).item()), 1.0)
    return float(torch.linalg.vector_norm(residual).item()) / denominator


class PackedExactGramBackend(AnalyticRidgeBackend):
    """Exact FP32 Gram state stored as unique upper-triangular blocks.

    Off-diagonal blocks are dense and diagonal blocks store only their upper
    triangle.  A dense symmetric workspace is reconstructed for each solve and
    discarded before returning from ``update``.
    """

    def __init__(self, *, block_size: int = 256, **kwargs) -> None:
        super().__init__(**kwargs)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = int(block_size)
        self._blocks: dict[tuple[int, int], torch.Tensor] = {}
        block_count = math.ceil(self.dimension / self.block_size)
        for row_block in range(block_count):
            row_start = row_block * self.block_size
            rows = min(row_start + self.block_size, self.dimension) - row_start
            for column_block in range(row_block, block_count):
                column_start = column_block * self.block_size
                columns = (
                    min(column_start + self.block_size, self.dimension)
                    - column_start
                )
                if row_block == column_block:
                    count = rows * (rows + 1) // 2
                    shape = (count,)
                else:
                    shape = (rows, columns)
                self._blocks[(row_block, column_block)] = torch.zeros(
                    shape, device=self.device, dtype=self.statistics_dtype
                )
        self.diagnostics.update(
            method="packed_exact_gram", block_size=self.block_size
        )

    def _block_bounds(self, block_index: int) -> tuple[int, int]:
        start = block_index * self.block_size
        return start, min(start + self.block_size, self.dimension)

    @staticmethod
    def _diagonal_indices(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu_indices(size, size, device=device)

    def _add_dense_symmetric(self, update: torch.Tensor) -> None:
        for (row_block, column_block), stored in self._blocks.items():
            row_start, row_stop = self._block_bounds(row_block)
            column_start, column_stop = self._block_bounds(column_block)
            block = update[row_start:row_stop, column_start:column_stop]
            if row_block == column_block:
                indices = self._diagonal_indices(len(block), block.device)
                stored.add_(block[indices[0], indices[1]])
            else:
                stored.add_(block)

    def _dense_gram(self, *, dtype: torch.dtype) -> torch.Tensor:
        dense = torch.zeros(
            (self.dimension, self.dimension), device=self.device, dtype=dtype
        )
        for (row_block, column_block), stored in self._blocks.items():
            row_start, row_stop = self._block_bounds(row_block)
            column_start, column_stop = self._block_bounds(column_block)
            if row_block == column_block:
                size = row_stop - row_start
                indices = self._diagonal_indices(size, dense.device)
                block = torch.zeros((size, size), device=dense.device, dtype=dtype)
                values = stored.to(dtype)
                block[indices[0], indices[1]] = values
                block = block + block.T
                block.diagonal().mul_(0.5)
                dense[row_start:row_stop, row_start:row_stop] = block
            else:
                values = stored.to(dtype)
                dense[row_start:row_stop, column_start:column_stop] = values
                dense[column_start:column_stop, row_start:row_stop] = values.T
        return dense

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        values, target_labels = self._validated_batch(features, labels)
        class_ids, cross, counts, targets = self._expanded_statistics(target_labels)
        gram_update = values.T @ values
        self._add_dense_symmetric(gram_update)
        del gram_update
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        system = self._dense_gram(dtype=self.solver_dtype)
        system.diagonal().add_(self.ridge_lambda)
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info.max().item()) != 0:
            raise RuntimeError("Packed Exact Ridge system is not positive definite")
        work_cross = new_cross.to(self.solver_dtype)
        weights = torch.cholesky_solve(work_cross, factor)
        residual_matrix = system @ weights - work_cross
        denominator = max(float(torch.linalg.vector_norm(work_cross).item()), 1.0)
        residual = float(torch.linalg.vector_norm(residual_matrix).item()) / denominator
        self.class_ids, self.Q, self.counts = class_ids, new_cross, new_counts
        self.weights = weights
        self.total_rows += len(values)
        self.diagnostics.update(
            solver_relative_residual=residual,
            total_rows=self.total_rows,
            packed_elements=packed_upper_element_count(self.dimension),
        )
        self.assert_exemplar_free_state()

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = self._common_persistent_tensors()
        for (row_block, column_block), values in self._blocks.items():
            tensors[f"gram.block_{row_block}_{column_block}"] = values
        return tensors

    def state_dict(self) -> dict[str, object]:
        state = self._common_state()
        state.update(
            method="packed_exact_gram_analytic_ridge",
            block_size=self.block_size,
            blocks={
                f"{row}_{column}": values.detach().cpu().clone()
                for (row, column), values in self._blocks.items()
            },
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        if state.get("method") != "packed_exact_gram_analytic_ridge":
            raise ValueError("invalid packed Exact checkpoint")
        if int(state.get("block_size", -1)) != self.block_size:
            raise ValueError("packed Exact block-size mismatch")
        self._load_common_values(state, dimension_field="dimension")
        encoded = state.get("blocks")
        expected = {f"{row}_{column}" for row, column in self._blocks}
        if not isinstance(encoded, dict) or set(encoded) != expected:
            raise ValueError("packed Exact checkpoint block inventory mismatch")
        for key, values in encoded.items():
            row, column = map(int, key.split("_"))
            target = self._blocks[(row, column)]
            loaded = values.to(device=self.device, dtype=self.statistics_dtype)
            if loaded.shape != target.shape or not bool(torch.isfinite(loaded).all()):
                raise ValueError("invalid packed Exact block")
            self._blocks[(row, column)] = loaded
        if self.total_rows:
            system = self._dense_gram(dtype=self.solver_dtype)
            system.diagonal().add_(self.ridge_lambda)
            factor, info = torch.linalg.cholesky_ex(system)
            if int(info.max().item()) != 0:
                raise ValueError("packed Exact checkpoint is not positive definite")
            work_cross = self.Q.to(self.solver_dtype)
            self.weights = torch.cholesky_solve(work_cross, factor)
            residual_matrix = system @ self.weights - work_cross
            denominator = max(float(torch.linalg.vector_norm(work_cross).item()), 1.0)
            self.diagnostics["solver_relative_residual"] = float(
                torch.linalg.vector_norm(residual_matrix).item()
            ) / denominator
        else:
            self.weights = None
        self.assert_exemplar_free_state()


class FrequentDirectionsRidgeBackend(AnalyticRidgeBackend):
    """Deterministic Frequent Directions sketch followed by a Ridge solve."""

    def __init__(self, *, sketch_rank: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if sketch_rank <= 0 or sketch_rank > self.dimension:
            raise ValueError("sketch_rank must be in [1, dimension]")
        self.sketch_rank = int(sketch_rank)
        self.sketch = torch.empty(
            (0, self.dimension), device=self.device, dtype=self.statistics_dtype
        )
        self.compression_count = 0
        self.covariance_error_bound = 0.0
        self.correction = torch.empty(
            (0, 0), device=self.device, dtype=torch.float64
        )
        self.diagnostics.update(
            method="frequent_directions_ridge",
            sketch_rank=self.sketch_rank,
            shrink_rule="delta_sigma_rank_squared",
        )

    def _compress(self, rows: torch.Tensor) -> tuple[torch.Tensor, float]:
        if rows.ndim != 2 or rows.shape[1] != self.dimension or not len(rows):
            raise ValueError("invalid Frequent Directions compression rows")
        _, singular_values, right = torch.linalg.svd(rows, full_matrices=False)
        keep = min(self.sketch_rank, int(singular_values.numel()))
        # Standard Frequent Directions shrinkage: when more than ell rows are
        # present, subtract sigma_ell^2 from the leading ell squared singular
        # values.  The last retained direction becomes zero and the sum of
        # these deltas certifies the covariance approximation error.
        delta = (
            float(singular_values[self.sketch_rank - 1].square().item())
            if singular_values.numel() > self.sketch_rank
            else 0.0
        )
        shrunk = singular_values[:keep].square().sub(delta).clamp_min_(0).sqrt_()
        sketch = shrunk[:, None] * right[:keep]
        self.compression_count += 1
        self.covariance_error_bound += delta
        return sketch.to(self.statistics_dtype), delta

    def _update_sketch(self, values: torch.Tensor) -> float:
        pending = values
        last_delta = 0.0
        while len(pending):
            capacity = max(1, 2 * self.sketch_rank - len(self.sketch))
            take = min(capacity, len(pending))
            combined = torch.cat((self.sketch, pending[:take]), dim=0)
            pending = pending[take:]
            if len(combined) > self.sketch_rank or not len(pending):
                self.sketch, last_delta = self._compress(combined)
            else:
                self.sketch = combined
        return last_delta

    def _solve(self, cross: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Solve through an FP64 compact Woodbury coefficient.

        The dense expression ``Q - B.T @ (...)`` suffers severe FP32
        cancellation on the 10k-wide, strongly regularized control.  We retain
        the small coefficient instead; its FP64 storage is accounted for.
        """

        sketch = self.sketch.to(torch.float64)
        work_cross = cross.to(torch.float64)
        ridge = float(self.ridge_lambda)
        if not len(sketch):
            correction = torch.empty(
                (0, work_cross.shape[1]), device=self.device, dtype=torch.float64
            )
            residual_matrix = torch.zeros_like(work_cross)
        else:
            core = sketch @ sketch.T
            core.diagonal().add_(ridge)
            factor, info = torch.linalg.cholesky_ex((core + core.T) * 0.5)
            if int(info.max().item()) != 0:
                raise RuntimeError("Frequent Directions Woodbury core is not SPD")
            projected = sketch @ work_cross
            correction = torch.cholesky_solve(projected, factor) / ridge
            # A(Q/lambda - B.T C)-Q =
            # B.T(BQ/lambda - (lambda I + BB.T)C).
            # This is the residual of the implicit classifier without
            # materializing a dense 10k-by-class weight matrix.
            residual_matrix = sketch.T @ (
                projected / ridge - core @ correction
            )
        denominator = max(float(torch.linalg.vector_norm(work_cross).item()), 1.0)
        residual = float(torch.linalg.vector_norm(residual_matrix).item()) / denominator
        return correction, residual

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        values, target_labels = self._validated_batch(features, labels)
        class_ids, cross, counts, targets = self._expanded_statistics(target_labels)
        last_delta = self._update_sketch(values)
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        correction, residual = self._solve(new_cross)
        self.class_ids, self.Q, self.counts = class_ids, new_cross, new_counts
        self.weights = None
        self.correction = correction
        self.total_rows += len(values)
        self.diagnostics.update(
            solver_relative_residual=residual,
            total_rows=self.total_rows,
            effective_rank=int(self.sketch.shape[0]),
            compression_count=self.compression_count,
            last_shrinkage_delta=last_delta,
            covariance_error_bound=self.covariance_error_bound,
        )
        if not bool(
            torch.isfinite(self.sketch).all()
            and torch.isfinite(self.correction).all()
        ):
            raise RuntimeError("Frequent Directions produced NaN or Inf")
        self.assert_exemplar_free_state()

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "Q": self.Q,
            "counts": self.counts,
            "fd_sketch": self.sketch,
            "woodbury_correction": self.correction,
        }

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        if not self.total_rows:
            raise RuntimeError("backend has not been updated")
        values = features.to(device=self.device, dtype=torch.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"features must have shape (B, {self.dimension})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        logits = values @ self.Q.to(torch.float64) / self.ridge_lambda
        if len(self.sketch):
            logits.sub_(
                (values @ self.sketch.to(torch.float64).T) @ self.correction
            )
        return logits

    def assert_exemplar_free_state(self) -> None:
        """Validate that the retained rows are an FD summary, not exemplars.

        The generic backend check rejects any tensor axis that happens to equal
        ``total_rows``.  That heuristic is appropriate for Gram/factor states,
        but it produces a false positive before an FD sketch has filled: the
        number of orthogonal summary rows can then equal the number of examples
        observed.  ``_update_sketch`` always passes the combined rows through
        an SVD before they become persistent state, so validation here is based
        on the defining fixed-rank shape rather than that accidental equality.
        """

        if self.Q.shape != (self.dimension, len(self.class_ids)):
            raise AssertionError("invalid Q shape")
        if self.counts.shape != (len(self.class_ids),):
            raise AssertionError("invalid class-count shape")
        if (
            self.sketch.ndim != 2
            or self.sketch.shape[1] != self.dimension
            or self.sketch.shape[0] > self.sketch_rank
        ):
            raise AssertionError("invalid Frequent Directions summary shape")
        if self.correction.shape != (len(self.sketch), len(self.class_ids)):
            raise AssertionError("invalid Frequent Directions correction shape")
        if self.total_rows > self.sketch_rank and len(self.sketch) != self.sketch_rank:
            raise AssertionError("filled Frequent Directions summary has wrong rank")
        if not bool(torch.isfinite(self.sketch).all()) or not bool(
            torch.isfinite(self.correction).all()
        ):
            raise AssertionError("non-finite Frequent Directions state")
        forbidden = ("history", "sample", "feature_cache", "labels", "codes")
        for name in self.persistent_tensors():
            if any(token in name.lower() for token in forbidden):
                raise AssertionError(f"forbidden sample-level state: {name}")

    def state_dict(self) -> dict[str, object]:
        state = self._common_state()
        state.update(
            method="frequent_directions_analytic_ridge",
            sketch_rank=self.sketch_rank,
            sketch=self.sketch.detach().cpu().clone(),
            compression_count=self.compression_count,
            covariance_error_bound=self.covariance_error_bound,
            correction=self.correction.detach().cpu().clone(),
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        if state.get("method") != "frequent_directions_analytic_ridge":
            raise ValueError("invalid Frequent Directions checkpoint")
        if int(state.get("sketch_rank", -1)) != self.sketch_rank:
            raise ValueError("Frequent Directions rank mismatch")
        self._load_common_values(state, dimension_field="dimension")
        sketch = state["sketch"].to(
            device=self.device, dtype=self.statistics_dtype
        )
        if (
            sketch.ndim != 2
            or sketch.shape[1] != self.dimension
            or len(sketch) > self.sketch_rank
            or not bool(torch.isfinite(sketch).all())
        ):
            raise ValueError("invalid Frequent Directions sketch")
        self.sketch = sketch
        self.compression_count = int(state.get("compression_count", 0))
        self.covariance_error_bound = float(
            state.get("covariance_error_bound", 0.0)
        )
        if self.compression_count < 0 or self.covariance_error_bound < 0:
            raise ValueError("invalid Frequent Directions compression count")
        correction = state["correction"].to(device=self.device, dtype=torch.float64)
        if correction.shape != (len(self.sketch), len(self.class_ids)) or not bool(
            torch.isfinite(correction).all()
        ):
            raise ValueError("invalid Frequent Directions Woodbury correction")
        self.correction = correction
        if self.total_rows:
            expected, residual = self._solve(self.Q)
            if not torch.allclose(expected, self.correction, atol=1e-10, rtol=1e-10):
                raise ValueError("Frequent Directions correction checkpoint mismatch")
            self.diagnostics["solver_relative_residual"] = residual
        else:
            self.correction = torch.empty((0, 0), device=self.device, dtype=torch.float64)
        self.assert_exemplar_free_state()

    def persistent_state_bytes(self) -> int:
        return persistent_tensor_bytes(self.persistent_tensors())


__all__ = [
    "FrequentDirectionsRidgeBackend",
    "PackedExactGramBackend",
    "frequent_directions_backend_bytes",
    "largest_frequent_directions_rank",
    "largest_packed_exact_dimension",
    "packed_exact_backend_bytes",
    "packed_upper_element_count",
]
