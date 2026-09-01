"""Exemplar-free learners for the SRQ-FLY D0 diagnostic."""

from __future__ import annotations

from contextlib import contextmanager
import inspect
import time

import torch

from models.flyhash import FlyHash

from .storage import CompressedUpper


def _tensor_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    if tensor.layout == torch.sparse_csc:
        return (
            tensor.values().numel() * tensor.values().element_size()
            + tensor.ccol_indices().numel() * tensor.ccol_indices().element_size()
            + tensor.row_indices().numel() * tensor.row_indices().element_size()
        )
    raise ValueError(f"unsupported tensor layout: {tensor.layout}")


def _relative_residual(system: torch.Tensor, weights: torch.Tensor, cross: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(system @ weights - cross)
    denominator = max(float(torch.linalg.vector_norm(cross).item()), 1.0)
    return float(numerator.item()) / denominator


def _relative_factor_residual(
    factor: torch.Tensor, weights: torch.Tensor, cross: torch.Tensor
) -> float:
    """Residual of ``(R.T @ R) W = Q`` without materializing ``R.T @ R``."""
    numerator = torch.linalg.vector_norm(factor.T @ (factor @ weights) - cross)
    denominator = max(float(torch.linalg.vector_norm(cross).item()), 1.0)
    return float(numerator.item()) / denominator


def _cholesky_solve(system: torch.Tensor, cross: torch.Tensor) -> tuple[torch.Tensor, float]:
    symmetric = (system + system.T) * 0.5
    factor, info = torch.linalg.cholesky_ex(symmetric)
    if int(info.max().item()) != 0:
        raise RuntimeError("compressed Ridge system is not numerically positive definite")
    weights = torch.cholesky_solve(cross, factor)
    return weights, _relative_residual(symmetric, weights, cross)


def _blocked_qr_rank_update(
    upper: torch.Tensor,
    update_rows: torch.Tensor,
    *,
    panel_size: int,
    trailing_chunk_size: int | None = None,
    preserve_update_rows: bool = True,
) -> torch.Tensor:
    """Return the positive-diagonal QR factor of ``[upper; update_rows]``.

    ``upper`` is already triangular.  A generic QR of the full stacked matrix
    repeats cubic work on its zero lower triangle.  This routine eliminates one
    column panel at a time using compact Householder reflectors.  At a panel it
    touches only the corresponding rows of ``upper`` and the (usually much
    smaller) rank-update rows.  Its leading work is therefore proportional to
    ``(rank + panel_size) * dimension**2`` rather than ``dimension**3``.

    The input factor is disposable after a streaming update, so it is reused as
    the output buffer.  ``update_rows`` is cloned by default because public
    callers still own their code tensor.  An explicitly consuming update may
    set ``preserve_update_rows=False`` after updating the cross statistic.
    When
    ``trailing_chunk_size`` is set, Householder reflectors are applied to only
    that many trailing columns at a time.  This leaves the arithmetic and
    factor contract unchanged while bounding the two temporary trailing
    matrices created by ``cat`` and ``ormqr``.
    """
    if upper.ndim != 2 or upper.shape[0] != upper.shape[1] or not len(upper):
        raise ValueError("upper must be a non-empty square matrix")
    if update_rows.ndim != 2 or update_rows.shape[1] != len(upper):
        raise ValueError("update rows must align with the factor dimension")
    if not len(update_rows) or panel_size <= 0:
        raise ValueError("rank and panel size must be positive")
    if trailing_chunk_size is not None and trailing_chunk_size <= 0:
        raise ValueError("trailing chunk size must be positive when provided")
    if upper.device != update_rows.device or upper.dtype != update_rows.dtype:
        raise ValueError("factor and update rows must share device and dtype")

    dimension = len(upper)
    residual = update_rows.clone() if preserve_update_rows else update_rows
    for start in range(0, dimension, panel_size):
        end = min(start + panel_size, dimension)
        width = end - start
        panel = torch.cat(
            (upper[start:end, start:end], residual[:, start:end]), dim=0
        )
        reflectors, tau = torch.geqrf(panel)
        diagonal_block = torch.triu(reflectors[:width])

        # QR is unique only up to row signs.  Positive diagonal factors keep
        # the compressed square-root contract and make repeated runs stable.
        signs = torch.where(
            diagonal_block.diagonal() < 0,
            -torch.ones((), device=upper.device, dtype=upper.dtype),
            torch.ones((), device=upper.device, dtype=upper.dtype),
        )

        trailing_width = (
            dimension - end
            if trailing_chunk_size is None
            else trailing_chunk_size
        )
        if end < dimension:
            for column_start in range(end, dimension, trailing_width):
                column_end = min(column_start + trailing_width, dimension)
                trailing = torch.cat(
                    (
                        upper[start:end, column_start:column_end],
                        residual[:, column_start:column_end],
                    ),
                    dim=0,
                )
                transformed = torch.ormqr(
                    reflectors, tau, trailing, left=True, transpose=True
                )
                upper[start:end, column_start:column_end].copy_(
                    signs[:, None] * transformed[:width]
                )
                residual[:, column_start:column_end].copy_(transformed[width:])

        upper[start:end, start:end].copy_(signs[:, None] * diagonal_block)
    return upper


class _BaseFLYLearner:
    is_exemplar_free = True

    def __init__(
        self,
        *,
        feature_dim: int,
        expand_dim: int,
        synaptic_degree: int,
        coding_level: float,
        ridge_lambda: float,
        block_size: int,
        group_size: int,
        seed: int = 2025,
        device: str | torch.device = "cpu",
        statistics_dtype: torch.dtype = torch.float32,
        solver_dtype: torch.dtype = torch.float32,
        projection: torch.Tensor | None = None,
        profile_updates: bool = False,
    ) -> None:
        if min(feature_dim, expand_dim, synaptic_degree, block_size, group_size) <= 0:
            raise ValueError("dimensions and storage groups must be positive")
        if synaptic_degree > feature_dim or not 0 < coding_level <= 1:
            raise ValueError("invalid FLY representation")
        if ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive")
        if statistics_dtype not in {torch.float32, torch.float64}:
            raise ValueError("invalid statistics dtype")
        if solver_dtype not in {torch.float32, torch.float64}:
            raise ValueError("invalid solver dtype")
        self.feature_dim = int(feature_dim)
        self.expand_dim = int(expand_dim)
        self.synaptic_degree = int(synaptic_degree)
        self.coding_level = float(coding_level)
        self.ridge_lambda = float(ridge_lambda)
        self.block_size = int(block_size)
        self.group_size = int(group_size)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.statistics_dtype = statistics_dtype
        self.solver_dtype = solver_dtype
        self.profile_updates = bool(profile_updates)

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

        self.Q = torch.zeros(
            (self.expand_dim, 0), device=self.device, dtype=self.statistics_dtype
        )
        self.counts = torch.zeros(0, device=self.device, dtype=self.statistics_dtype)
        self.class_ids: list[int] = []
        self.weights: torch.Tensor | None = None
        self.total_rows = 0
        self.diagnostics: dict = {
            "task_id_required": False,
            "ridge_lambda": self.ridge_lambda,
        }

    @contextmanager
    def _profile_stage(
        self,
        name: str,
        timings: dict[str, float],
        memory: dict[str, dict[str, int | None]],
    ):
        """Measure one stage in a separate opt-in profiling run.

        Per-stage CUDA counters are deliberately disjoint from the ordinary
        whole-update measurement.  Callers therefore profile a fresh learner
        only after the timed worker has completed.
        """
        if not self.profile_updates:
            yield
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            before_allocated = int(torch.cuda.memory_allocated(self.device))
            before_reserved = int(torch.cuda.memory_reserved(self.device))
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            before_allocated = before_reserved = None
        started = time.perf_counter()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings[name] = time.perf_counter() - started
        if self.device.type == "cuda":
            memory[name] = {
                "before_allocated_bytes": before_allocated,
                "after_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "before_reserved_bytes": before_reserved,
                "after_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
            }
        else:
            memory[name] = {
                "before_allocated_bytes": None,
                "after_allocated_bytes": None,
                "peak_allocated_bytes": None,
                "before_reserved_bytes": None,
                "after_reserved_bytes": None,
                "peak_reserved_bytes": None,
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

    def _expanded_statistics(self, labels: torch.Tensor):
        updated = sorted(set(self.class_ids) | set(map(int, labels.cpu().tolist())))
        old_columns = {value: index for index, value in enumerate(self.class_ids)}
        new_columns = {value: index for index, value in enumerate(updated)}
        cross = torch.zeros(
            (self.expand_dim, len(updated)), device=self.device, dtype=self.statistics_dtype
        )
        counts = torch.zeros(len(updated), device=self.device, dtype=self.statistics_dtype)
        for class_id, old_column in old_columns.items():
            column = new_columns[class_id]
            cross[:, column] = self.Q[:, old_column]
            counts[column] = self.counts[old_column]
        columns = torch.tensor(
            [new_columns[int(value)] for value in labels.cpu().tolist()],
            device=self.device,
            dtype=torch.long,
        )
        targets = torch.nn.functional.one_hot(columns, num_classes=len(updated)).to(
            self.statistics_dtype
        )
        return updated, cross, counts, targets

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.update_codes(self.encode(features), labels)

    def predict_logits_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("learner has not been updated")
        values = codes.to(device=self.device, dtype=self.weights.dtype)
        if values.ndim != 2 or values.shape[1] != self.expand_dim:
            raise ValueError(f"codes must have shape (B, {self.expand_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("codes contain NaN or Inf")
        return values @ self.weights

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.predict_logits_from_codes(self.encode(features))

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(features).argmax(1).cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns])

    def _base_persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "projection": self.flyhash.projection_matrix,
            "Q": self.Q,
            "counts": self.counts,
        }
        if self.weights is not None:
            tensors["weights"] = self.weights
        return tensors

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def persistent_state_bytes(self) -> int:
        return sum(_tensor_bytes(value) for value in self.persistent_tensors().values())

    def assert_exemplar_free_state(self) -> None:
        if self.Q.shape != (self.expand_dim, len(self.class_ids)):
            raise AssertionError("invalid Q shape")
        if self.counts.shape != (len(self.class_ids),):
            raise AssertionError("invalid class-count shape")
        if self.weights is not None and self.weights.shape != self.Q.shape:
            raise AssertionError("invalid classifier shape")
        forbidden = ("history", "sample", "feature_cache", "labels", "codes")
        for name, tensor in self.persistent_tensors().items():
            if any(token in name.lower() for token in forbidden):
                raise AssertionError(f"forbidden sample-level state: {name}")
            if tensor.ndim >= 2 and self.total_rows not in {
                self.feature_dim, self.expand_dim, len(self.class_ids)
            } and self.total_rows in tensor.shape:
                raise AssertionError(f"historical sample dimension in {name}")

    def _configuration_state(self) -> dict:
        return {
            "version": 1,
            "feature_dim": self.feature_dim,
            "expand_dim": self.expand_dim,
            "synaptic_degree": self.synaptic_degree,
            "coding_level": self.coding_level,
            "ridge_lambda": self.ridge_lambda,
            "block_size": self.block_size,
            "group_size": self.group_size,
            "seed": self.seed,
            "statistics_dtype": str(self.statistics_dtype).removeprefix("torch."),
            "solver_dtype": str(self.solver_dtype).removeprefix("torch."),
            "projection": self.flyhash.projection_matrix.detach().cpu(),
            "Q": self.Q.detach().cpu().clone(),
            "counts": self.counts.detach().cpu().clone(),
            "class_ids": list(self.class_ids),
            "total_rows": self.total_rows,
        }

    def _load_common(self, state: dict, expected_method: str) -> None:
        if state.get("version") != 1 or state.get("method") != expected_method:
            raise ValueError("unsupported SRQ-FLY checkpoint")
        for field in (
            "feature_dim", "expand_dim", "synaptic_degree", "coding_level",
            "ridge_lambda", "block_size", "group_size", "seed",
        ):
            if state.get(field) != getattr(self, field):
                raise ValueError(f"checkpoint configuration mismatch for {field}")
        if state.get("statistics_dtype") != str(self.statistics_dtype).removeprefix("torch."):
            raise ValueError("checkpoint statistics dtype mismatch")
        if state.get("solver_dtype") != str(self.solver_dtype).removeprefix("torch."):
            raise ValueError("checkpoint solver dtype mismatch")
        projection = state["projection"].to(self.device)
        if projection.layout != torch.sparse_csc or projection.shape != (
            self.expand_dim, self.feature_dim
        ) or not bool(torch.isfinite(projection.values()).all()):
            raise ValueError("invalid checkpoint projection")
        self.flyhash.projection_matrix = projection
        self.Q = state["Q"].to(device=self.device, dtype=self.statistics_dtype)
        self.counts = state["counts"].to(device=self.device, dtype=self.statistics_dtype)
        self.class_ids = [int(value) for value in state["class_ids"]]
        self.total_rows = int(state["total_rows"])
        if self.class_ids != sorted(set(self.class_ids)):
            raise ValueError("checkpoint class IDs must be sorted and unique")
        if self.total_rows < 0 or self.Q.shape != (
            self.expand_dim, len(self.class_ids)
        ) or self.counts.shape != (len(self.class_ids),):
            raise ValueError("invalid checkpoint sufficient-statistic shapes")
        if not bool(torch.isfinite(self.Q).all()) or not bool(torch.isfinite(self.counts).all()):
            raise ValueError("checkpoint sufficient statistics contain NaN or Inf")
        if bool((self.counts < 0).any()) or abs(float(self.counts.sum()) - self.total_rows) > 1e-3:
            raise ValueError("checkpoint counts do not match total_rows")

    def _validate_compressed(self, value: CompressedUpper, *, mode: str) -> None:
        if (
            value.dimension != self.expand_dim
            or value.block_size != self.block_size
            or value.group_size != self.group_size
            or value.mode != mode
        ):
            raise ValueError("compressed checkpoint configuration mismatch")


class DirectInt8GramLearner(_BaseFLYLearner):
    """Groupwise-int8 full Gram without a worst-case certificate gate."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gram: CompressedUpper | None = None
        self.diagnostics.update(method="direct_int8_gram", storage="groupwise_int8")

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
            updated = updated + self.gram.reconstruct_symmetric(dtype=self.statistics_dtype)
        compressed = CompressedUpper.from_upper(
            updated,
            block_size=self.block_size,
            group_size=self.group_size,
            mode="int8",
        )
        reconstructed = compressed.reconstruct_symmetric(dtype=self.solver_dtype)
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        work_cross = new_cross.to(self.solver_dtype)
        system = reconstructed + self.ridge_lambda * torch.eye(
            self.expand_dim, device=self.device, dtype=self.solver_dtype
        )
        weights, residual = _cholesky_solve(system, work_cross)
        relative_storage_error = float(
            torch.linalg.vector_norm(reconstructed.to(updated.dtype) - updated).item()
        ) / max(float(torch.linalg.vector_norm(updated).item()), 1.0)

        self.gram = compressed
        self.class_ids, self.Q, self.counts = class_ids, new_cross, new_counts
        self.weights = weights
        self.total_rows += len(values)
        self.diagnostics.update(
            solver_relative_residual=residual,
            relative_local_storage_error=relative_storage_error,
            total_rows=self.total_rows,
        )
        self.assert_exemplar_free_state()

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = self._base_persistent_tensors()
        if self.gram is not None:
            tensors.update(self.gram.persistent_tensors("gram"))
        return tensors

    def state_dict(self) -> dict:
        state = self._configuration_state()
        state.update(
            method="direct_int8_gram",
            gram=None if self.gram is None else self.gram.state_dict(),
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        self._load_common(state, "direct_int8_gram")
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
            system = reconstructed + self.ridge_lambda * torch.eye(
                self.expand_dim, device=self.device, dtype=self.solver_dtype
            )
            self.weights, residual = _cholesky_solve(system, self.Q.to(self.solver_dtype))
            self.diagnostics["solver_relative_residual"] = residual
        self.assert_exemplar_free_state()


class SquareRootFLYLearner(_BaseFLYLearner):
    """Full-rank square-root FLY with float16 or groupwise-int8 storage."""

    def __init__(
        self,
        *,
        storage_mode: str,
        update_backend: str = "gram_cholesky",
        update_panel_size: int = 128,
        update_trailing_chunk_size: int | None = None,
        quantization_backend: str = "eager",
        quantization_batch_blocks: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if storage_mode not in {"float16", "int8"}:
            raise ValueError("storage_mode must be float16 or int8")
        if update_backend not in {
            "gram_cholesky", "gram_cholesky_direct", "stacked_qr", "blocked_qr"
        }:
            raise ValueError(
                "update_backend must be gram_cholesky, "
                "gram_cholesky_direct, stacked_qr, or blocked_qr"
            )
        if update_panel_size <= 0:
            raise ValueError("update_panel_size must be positive")
        if update_trailing_chunk_size is not None and update_trailing_chunk_size <= 0:
            raise ValueError("update_trailing_chunk_size must be positive")
        if quantization_backend not in {"eager", "streaming"}:
            raise ValueError("quantization_backend must be eager or streaming")
        if quantization_batch_blocks <= 0:
            raise ValueError("quantization_batch_blocks must be positive")
        if quantization_backend == "streaming" and update_backend not in {
            "gram_cholesky_direct", "blocked_qr"
        }:
            raise ValueError(
                "streaming quantization requires an in-place update backend"
            )
        self.storage_mode = storage_mode
        self.update_backend = update_backend
        self.update_panel_size = int(update_panel_size)
        self.update_trailing_chunk_size = (
            None
            if update_trailing_chunk_size is None
            else int(update_trailing_chunk_size)
        )
        self.quantization_backend = quantization_backend
        self.quantization_batch_blocks = int(quantization_batch_blocks)
        self.factor: CompressedUpper | None = None
        self.diagnostics.update(
            method="sqrt_float16" if storage_mode == "float16" else "srq_int8",
            storage=storage_mode,
            structurally_spd=True,
            update_backend=update_backend,
            update_panel_size=self.update_panel_size,
            update_trailing_chunk_size=self.update_trailing_chunk_size,
            quantization_backend=self.quantization_backend,
            quantization_batch_blocks=self.quantization_batch_blocks,
        )

    def update_codes(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        self._update_codes(codes, labels, consume_codes=False)

    def update_codes_consuming(
        self, codes: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Update while treating ``codes`` as a disposable work buffer.

        This opt-in path is intended for isolated experiment runners that
        materialize a code tensor solely for one streaming update.  The public
        ``update_codes`` method retains its non-mutating input contract.
        """
        if self.update_backend != "blocked_qr":
            raise ValueError("consuming updates require the blocked_qr backend")
        self._update_codes(codes, labels, consume_codes=True)

    def _update_codes(
        self, codes: torch.Tensor, labels: torch.Tensor, *, consume_codes: bool
    ) -> None:
        update_started = time.perf_counter()
        timings: dict[str, float] = {}
        memory: dict[str, dict[str, int | None]] = {}
        values = codes.to(device=self.device, dtype=self.statistics_dtype)
        target_labels = labels.to(device=self.device, dtype=torch.long)
        if values.ndim != 2 or values.shape[1] != self.expand_dim:
            raise ValueError(f"codes must have shape (B, {self.expand_dim})")
        if target_labels.ndim != 1 or len(target_labels) != len(values) or not len(values):
            raise ValueError("labels must align with a non-empty code matrix")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("codes contain NaN or Inf")
        with self._profile_stage("class_expansion", timings, memory):
            class_ids, cross, counts, targets = self._expanded_statistics(target_labels)
        new_cross = new_counts = work_cross = None
        if consume_codes:
            with self._profile_stage("cross_update", timings, memory):
                new_cross = cross + values.T @ targets
                new_counts = counts + targets.sum(0)
                work_cross = new_cross.to(self.solver_dtype)
        solve_values = values.to(self.solver_dtype)
        previous = None
        if self.factor is not None:
            with self._profile_stage("factor_decode", timings, memory):
                previous = self.factor.reconstruct_upper(dtype=self.solver_dtype)

        if previous is not None and self.update_backend == "blocked_qr":
            with self._profile_stage("blocked_qr", timings, memory):
                exact_upper = _blocked_qr_rank_update(
                    previous,
                    solve_values,
                    panel_size=self.update_panel_size,
                    trailing_chunk_size=self.update_trailing_chunk_size,
                    preserve_update_rows=not consume_codes,
                )
        elif previous is not None and self.update_backend == "stacked_qr":
            with self._profile_stage("stacked_qr", timings, memory):
                stacked = torch.cat((previous, solve_values), dim=0)
                _, exact_upper = torch.linalg.qr(stacked, mode="r")
                signs = torch.where(
                    exact_upper.diagonal() < 0,
                    -torch.ones((), device=self.device, dtype=self.solver_dtype),
                    torch.ones((), device=self.device, dtype=self.solver_dtype),
                )
                exact_upper = signs[:, None] * exact_upper
                del stacked
        else:
            with self._profile_stage("current_gram", timings, memory):
                updated_system = solve_values.T @ solve_values
            if previous is None:
                # Adding the ridge directly to the diagonal avoids allocating
                # a dense m-by-m identity matrix on every first update.
                updated_system.diagonal().add_(self.ridge_lambda)
            else:
                with self._profile_stage("previous_factor_product", timings, memory):
                    if self.update_backend == "gram_cholesky_direct":
                        # The fused GEMM writes directly into the current
                        # system buffer and avoids another m-by-m tensor.
                        updated_system.addmm_(previous.T, previous)
                    else:
                        # Retain the historical operation ordering for the
                        # bitwise-compatible checkpoint backend.
                        updated_system.add_(previous.T @ previous)
                del previous
            with self._profile_stage("cholesky", timings, memory):
                # Both terms used above are Gram products and therefore
                # symmetric by construction.  The historical backend keeps
                # the explicit average for bitwise compatibility.  The
                # opt-in direct backend lets Cholesky consume the lower
                # triangle in place and avoids another dense m-by-m
                # temporary.  It is eligible only after its predictor drift
                # is checked by the isolated system benchmark.
                cholesky_input = (
                    updated_system
                    if self.update_backend == "gram_cholesky_direct"
                    else (updated_system + updated_system.T) * 0.5
                )
                exact_lower, info = torch.linalg.cholesky_ex(cholesky_input)
                if int(info.max().item()) != 0:
                    raise RuntimeError("square-root streaming update failed Cholesky")
                exact_upper = exact_lower.T
                # The factor view owns the Cholesky output.  The dense Gram
                # and (for the compatibility path) symmetrized input are dead
                # before quantization and otherwise inflate its baseline by
                # up to two m-by-m floating-point matrices.
                del updated_system, cholesky_input, exact_lower

        with self._profile_stage("factor_quantization", timings, memory):
            if self.update_backend in {"gram_cholesky_direct", "blocked_qr"}:
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
                else:
                    compressed, relative_factor_error = CompressedUpper.from_upper_inplace(
                        exact_upper,
                        block_size=self.block_size,
                        group_size=self.group_size,
                        mode=self.storage_mode,
                    )
            else:
                compressed = CompressedUpper.from_upper(
                    exact_upper,
                    block_size=self.block_size,
                    group_size=self.group_size,
                    mode=self.storage_mode,
                )
        with self._profile_stage("factor_reconstruction", timings, memory):
            reconstructed = (
                exact_upper
                if self.update_backend in {"gram_cholesky_direct", "blocked_qr"}
                else compressed.reconstruct_upper(dtype=self.solver_dtype)
            )
        if bool((reconstructed.diagonal() <= 0).any()):
            raise RuntimeError("compressed square-root diagonal is not positive")
        if not consume_codes:
            with self._profile_stage("cross_update", timings, memory):
                new_cross = cross + values.T @ targets
                new_counts = counts + targets.sum(0)
                work_cross = new_cross.to(self.solver_dtype)
        if new_cross is None or new_counts is None or work_cross is None:
            raise RuntimeError("internal cross-statistic update failure")
        with self._profile_stage("triangular_solve", timings, memory):
            intermediate = torch.linalg.solve_triangular(
                reconstructed.T, work_cross, upper=False
            )
            weights = torch.linalg.solve_triangular(
                reconstructed, intermediate, upper=True
            )
        with self._profile_stage("diagnostics", timings, memory):
            residual = _relative_factor_residual(reconstructed, weights, work_cross)
            if self.update_backend not in {"gram_cholesky_direct", "blocked_qr"}:
                relative_factor_error = float(
                    torch.dist(reconstructed, exact_upper).item()
                ) / max(float(torch.linalg.vector_norm(exact_upper).item()), 1.0)

        self.factor = compressed
        self.class_ids, self.Q, self.counts = class_ids, new_cross, new_counts
        self.weights = weights
        self.total_rows += len(values)
        self.diagnostics.update(
            solver_relative_residual=residual,
            relative_local_factor_error=relative_factor_error,
            total_rows=self.total_rows,
        )
        if self.profile_updates:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.diagnostics["last_update_stage_seconds"] = timings
            self.diagnostics["last_update_stage_cuda_memory"] = memory
            self.diagnostics["last_update_total_seconds"] = (
                time.perf_counter() - update_started
            )
        self.assert_exemplar_free_state()

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = self._base_persistent_tensors()
        if self.factor is not None:
            tensors.update(self.factor.persistent_tensors("factor"))
        return tensors

    def state_dict(self) -> dict:
        state = self._configuration_state()
        state.update(
            method="square_root_fly",
            storage_mode=self.storage_mode,
            update_backend=self.update_backend,
            update_panel_size=self.update_panel_size,
            update_trailing_chunk_size=self.update_trailing_chunk_size,
            factor=None if self.factor is None else self.factor.state_dict(),
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        self._load_common(state, "square_root_fly")
        if state.get("storage_mode") != self.storage_mode:
            raise ValueError("checkpoint square-root storage mode mismatch")
        if state.get("update_backend", "gram_cholesky") != self.update_backend:
            raise ValueError("checkpoint square-root update backend mismatch")
        if state.get("update_panel_size", 128) != self.update_panel_size:
            raise ValueError("checkpoint update panel size mismatch")
        if state.get("update_trailing_chunk_size") != self.update_trailing_chunk_size:
            raise ValueError("checkpoint update trailing chunk size mismatch")
        self.factor = None if state["factor"] is None else CompressedUpper.load_state_dict(
            state["factor"], device=self.device
        )
        if self.factor is None:
            if self.total_rows or self.class_ids:
                raise ValueError("non-empty checkpoint is missing its square-root state")
            self.weights = None
        else:
            self._validate_compressed(self.factor, mode=self.storage_mode)
            reconstructed = self.factor.reconstruct_upper(dtype=self.solver_dtype)
            intermediate = torch.linalg.solve_triangular(
                reconstructed.T, self.Q.to(self.solver_dtype), upper=False
            )
            self.weights = torch.linalg.solve_triangular(
                reconstructed, intermediate, upper=True
            )
            residual = _relative_factor_residual(
                reconstructed, self.weights, self.Q.to(self.solver_dtype)
            )
            self.diagnostics["solver_relative_residual"] = residual
        self.assert_exemplar_free_state()


assert "task_id" not in inspect.signature(_BaseFLYLearner.predict_logits).parameters
