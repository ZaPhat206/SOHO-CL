"""Declared block-selection criteria for mixed INT8/FP16 factor storage.

The locked :meth:`AdaptiveCompressedUpper.from_upper_inplace` implements the
factor-MSE criterion behind every reported SRQ-Adaptive result, and it is not
modified here.  This module reproduces its encoding, byte accounting, greedy
selection, in-place decoding and diagnostics operation by operation, but lets
the per-block benefit come from a declared criterion:

``factor_mse``
    Reduction in squared reconstruction error of the factor.  Selections,
    stored tensors and decoded values are identical to the locked method.

``row_weighted_system``
    The same per-entry reduction, weighted by the squared norm of the matching
    row of the unquantized factor.  For a quantization error ``E`` whose rows
    are independent with zero mean,
    ``E ||R^T E||_F^2 = sum_i ||R_{i.}||^2 E ||E_{i.}||^2``, so the weighted
    reduction is the expected reduction of the first-order term of the
    per-task system error ``D_t = R^T E + E^T R + E^T E``.

The module also provides a batched computation of the factor-MSE benefits.
It is used only to benchmark scoring time and decision agreement; stored
state always comes from the per-block path.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Callable

import torch

from .adaptive_upper import (
    AdaptiveCompressedUpper,
    AdaptiveUpperBlock,
    _groupwise_int8,
)
from .backends import SquareRootBackend, _relative_factor_residual
from .qr import blocked_qr_rank_update


CRITERIA = ("factor_mse", "row_weighted_system")


def upper_block_descriptors(dimension: int, block_size: int) -> list[tuple[int, int]]:
    """Upper-triangular block coordinates in the locked storage order."""
    if dimension <= 0 or block_size <= 0:
        raise ValueError("dimension and block_size must be positive")
    count = math.ceil(dimension / block_size)
    return [(row, column) for row in range(count) for column in range(row, count)]


def _bounds(block: int, block_size: int, dimension: int) -> tuple[int, int]:
    start = block * block_size
    return start, min(start + block_size, dimension)


def _validate_matrix(matrix: torch.Tensor, block_size: int, group_size: int) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("matrix must be non-empty and square")
    if matrix.dtype not in {torch.float32, torch.float64}:
        raise ValueError("matrix must use float32 or float64")
    if min(block_size, group_size) <= 0:
        raise ValueError("block_size and group_size must be positive")


def factor_row_weights(matrix: torch.Tensor, *, chunk_rows: int = 1024) -> torch.Tensor:
    """Squared row norms of an upper factor, in float64, without a dense copy."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    parts = []
    for start in range(0, len(matrix), chunk_rows):
        chunk = matrix[start : start + chunk_rows].to(torch.float64)
        parts.append(chunk.square().sum(1))
    return torch.cat(parts)


def score_blocks(
    matrix: torch.Tensor,
    *,
    block_size: int,
    group_size: int,
    criterion: str,
    row_weights: torch.Tensor | None = None,
) -> dict:
    """Encode every strict-upper block both ways and score it.

    The loop, encodings and error reductions are the locked ones.  Only the
    returned ``benefits`` depend on ``criterion``.
    """
    _validate_matrix(matrix, block_size, group_size)
    if criterion not in CRITERIA:
        raise ValueError(f"unknown selection criterion {criterion!r}")
    dimension = len(matrix)
    if criterion == "row_weighted_system":
        if (
            row_weights is None
            or row_weights.shape != (dimension,)
            or row_weights.dtype != torch.float64
            or not bool(torch.isfinite(row_weights).all())
            or bool((row_weights < 0).any())
        ):
            raise ValueError("row_weighted_system needs finite non-negative float64 row weights")
        row_weights = row_weights.to(matrix.device)
    elif row_weights is not None:
        raise ValueError("row weights are only used by row_weighted_system")

    count = math.ceil(dimension / block_size)
    descriptors: list[tuple[int, int]] = []
    int8_blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
    int8_errors: list[torch.Tensor] = []
    fp16_errors: list[torch.Tensor] = []
    reference_sums: list[torch.Tensor] = []
    weighted_benefits: list[torch.Tensor] = []
    extra_costs: list[int] = []
    value_counts: list[int] = []

    for row_block in range(count):
        rs, re = _bounds(row_block, block_size, dimension)
        for col_block in range(row_block, count):
            cs, ce = _bounds(col_block, block_size, dimension)
            local = matrix[rs:re, cs:ce]
            if row_block == col_block:
                indices = torch.triu_indices(
                    re - rs, ce - cs, offset=1, device=matrix.device
                )
                source = local[indices[0], indices[1]]
            else:
                indices = None
                source = local.contiguous().view(-1)
            encoded, scales, decoded_int8 = _groupwise_int8(source, group_size)
            encoded_fp16 = source.to(torch.float16)
            if not bool(torch.isfinite(encoded_fp16).all()):
                raise ValueError("adaptive FP16 candidate overflowed")
            decoded_fp16 = encoded_fp16.to(source.dtype)
            descriptors.append((row_block, col_block))
            int8_blocks.append((encoded, scales))
            int8_entry_error = (decoded_int8 - source).square()
            fp16_entry_error = (decoded_fp16 - source).square()
            int8_errors.append(int8_entry_error.sum())
            fp16_errors.append(fp16_entry_error.sum())
            reference_sums.append(source.square().sum())
            if criterion == "row_weighted_system":
                if indices is None:
                    weights = row_weights[rs:re].repeat_interleave(ce - cs)
                else:
                    weights = row_weights[indices[0] + rs]
                reduction = (int8_entry_error - fp16_entry_error).to(torch.float64)
                weighted_benefits.append((weights * reduction).sum())
            entries = int(source.numel())
            int8_bytes = entries + 4 * math.ceil(entries / group_size)
            fp16_bytes = 2 * entries
            extra_costs.append(fp16_bytes - int8_bytes)
            value_counts.append(entries)

    if criterion == "factor_mse":
        benefit = torch.stack(int8_errors) - torch.stack(fp16_errors)
        benefits = benefit.detach().to(torch.float64).cpu().tolist()
    else:
        benefits = torch.stack(weighted_benefits).detach().cpu().tolist()
    return {
        "descriptors": descriptors,
        "int8_blocks": int8_blocks,
        "int8_errors": int8_errors,
        "fp16_errors": fp16_errors,
        "reference_sums": reference_sums,
        "benefits": benefits,
        "extra_costs": extra_costs,
        "value_counts": value_counts,
    }


def budget_bytes(
    value_counts: list[int], *, group_size: int, budget_fraction: float
) -> tuple[int, int, int]:
    """All-INT8 bytes, all-FP16 bytes and the extra-byte budget (locked rule)."""
    if not 0.0 <= float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must lie in [0, 1]")
    base = sum(entries + 4 * math.ceil(entries / group_size) for entries in value_counts)
    full = 2 * sum(value_counts)
    return base, full, math.floor(float(budget_fraction) * (full - base))


def greedy_select(
    benefits: list[float], extra_costs: list[int], extra_budget: int
) -> tuple[list[bool], int]:
    """Locked greedy rule: best benefit per byte first, skip what does not fit."""
    if len(benefits) != len(extra_costs):
        raise ValueError("benefits and costs must align")
    order = sorted(
        range(len(benefits)),
        key=lambda index: (-benefits[index] / max(extra_costs[index], 1), index),
    )
    selected = [False] * len(benefits)
    used_extra = 0
    for index in order:
        cost = extra_costs[index]
        if benefits[index] <= 0 or cost <= 0:
            continue
        if used_extra + cost <= extra_budget:
            selected[index] = True
            used_extra += cost
    return selected, used_extra


def compress_upper_with_criterion(
    matrix: torch.Tensor,
    *,
    block_size: int,
    group_size: int,
    budget_fraction: float,
    criterion: str,
    row_weights: torch.Tensor | None = None,
) -> tuple[AdaptiveCompressedUpper, float, dict[str, int | float | str]]:
    """Criterion-parameterized twin of the locked adaptive compressor.

    With ``criterion="factor_mse"`` every returned tensor, the in-place decoded
    matrix and every diagnostic equal those of
    :meth:`AdaptiveCompressedUpper.from_upper_inplace`.
    """
    _validate_matrix(matrix, block_size, group_size)
    if not 0.0 <= float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must lie in [0, 1]")
    finite = torch.ones((), device=matrix.device, dtype=torch.bool)
    validation_rows = max(block_size, min(len(matrix), 1024))
    for start in range(0, len(matrix), validation_rows):
        finite.logical_and_(torch.isfinite(matrix[start : start + validation_rows]).all())
    if not bool(finite):
        raise ValueError("matrix contains NaN or Inf")

    dimension = len(matrix)
    diagonal = matrix.diagonal().to(torch.float32).clone()
    scores = score_blocks(
        matrix,
        block_size=block_size,
        group_size=group_size,
        criterion=criterion,
        row_weights=row_weights,
    )
    descriptors = scores["descriptors"]
    value_counts = scores["value_counts"]
    base_strict_bytes, full_fp16_strict_bytes, extra_budget = budget_bytes(
        value_counts, group_size=group_size, budget_fraction=budget_fraction
    )
    selected, used_extra = greedy_select(
        scores["benefits"], scores["extra_costs"], extra_budget
    )

    blocks: list[AdaptiveUpperBlock] = []
    for index, (row_block, col_block) in enumerate(descriptors):
        rs, re = _bounds(row_block, block_size, dimension)
        cs, ce = _bounds(col_block, block_size, dimension)
        local = matrix[rs:re, cs:ce]
        if row_block == col_block:
            local_indices = torch.triu_indices(
                re - rs, ce - cs, offset=1, device=matrix.device
            )
            source = local[local_indices[0], local_indices[1]]
        else:
            source = local.contiguous().view(-1)
        if selected[index]:
            values = source.to(torch.float16)
            scales = None
            decoded = values.to(source.dtype)
        else:
            values, scales = scores["int8_blocks"][index]
            decoded = values.to(source.dtype) * scales.to(source.dtype).repeat_interleave(
                group_size
            )[: len(values)]
        blocks.append(AdaptiveUpperBlock(row_block, col_block, values, scales))
        if row_block == col_block:
            local[local_indices[0], local_indices[1]] = decoded
        else:
            local.copy_(decoded.reshape(re - rs, ce - cs))

    int8_errors = scores["int8_errors"]
    fp16_errors = scores["fp16_errors"]
    original_diagonal = matrix.diagonal().to(torch.float64)
    diagonal_error = (diagonal.to(torch.float64) - original_diagonal).square().sum()
    diagonal_reference = original_diagonal.square().sum()
    mask = torch.tensor(selected, device=matrix.device, dtype=torch.uint8)
    chosen_errors = torch.where(
        mask.to(torch.bool), torch.stack(fp16_errors), torch.stack(int8_errors)
    )
    squared_error = diagonal_error + chosen_errors.to(torch.float64).sum()
    squared_reference = diagonal_reference + torch.stack(scores["reference_sums"]).to(
        torch.float64
    ).sum()
    matrix.diagonal().copy_(diagonal.to(matrix.dtype))
    relative_error = float(
        torch.sqrt(squared_error / torch.clamp(squared_reference, min=1.0)).item()
    )
    all_int8_relative_error = float(
        torch.sqrt(
            (diagonal_error + torch.stack(int8_errors).to(torch.float64).sum())
            / torch.clamp(squared_reference, min=1.0)
        ).item()
    )
    all_fp16_relative_error = float(
        torch.sqrt(
            (diagonal_error + torch.stack(fp16_errors).to(torch.float64).sum())
            / torch.clamp(squared_reference, min=1.0)
        ).item()
    )
    state = AdaptiveCompressedUpper(
        dimension=dimension,
        block_size=block_size,
        group_size=group_size,
        budget_fraction=budget_fraction,
        diagonal=diagonal,
        precision_mask=mask,
        blocks=blocks,
        validate_values=False,
    )
    diagnostics: dict[str, int | float | str] = {
        "selected_fp16_blocks": sum(selected),
        "total_blocks": len(selected),
        "selected_fp16_values": sum(
            value_counts[index] for index, flag in enumerate(selected) if flag
        ),
        "total_strict_upper_values": sum(value_counts),
        "int8_strict_upper_bytes": base_strict_bytes,
        "fp16_strict_upper_bytes": full_fp16_strict_bytes,
        "extra_budget_bytes": extra_budget,
        "used_extra_bytes": used_extra,
        "factor_persistent_bytes": state.persistent_state_bytes(),
        "factor_budget_ceiling_bytes": (
            4 * dimension + len(selected) + base_strict_bytes + extra_budget
        ),
        "all_int8_relative_factor_error": all_int8_relative_error,
        "all_fp16_relative_factor_error": all_fp16_relative_error,
        "selection_criterion": criterion,
    }
    return state, relative_error, diagnostics


def batched_factor_mse_benefits(
    matrix: torch.Tensor, *, block_size: int, group_size: int, batch_blocks: int
) -> tuple[list[float], list[int], list[int]]:
    """Factor-MSE benefits with one gather and one encode per batch of blocks.

    Blocks of equal shape are gathered with a single advanced index and encoded
    together.  Per-entry arithmetic is the locked arithmetic; only the order of
    the final per-block summation may differ, so benefits can differ in their
    last floating-point bits.
    """
    _validate_matrix(matrix, block_size, group_size)
    if batch_blocks <= 0:
        raise ValueError("batch_blocks must be positive")
    dimension = len(matrix)
    device = matrix.device
    descriptors = upper_block_descriptors(dimension, block_size)
    shapes: dict[tuple[bool, int, int], list[int]] = defaultdict(list)
    extra_costs: list[int] = []
    value_counts: list[int] = []
    for index, (row_block, col_block) in enumerate(descriptors):
        rs, re = _bounds(row_block, block_size, dimension)
        cs, ce = _bounds(col_block, block_size, dimension)
        diagonal_block = row_block == col_block
        rows, columns = re - rs, ce - cs
        entries = rows * (rows - 1) // 2 if diagonal_block else rows * columns
        shapes[(diagonal_block, rows, columns)].append(index)
        extra_costs.append(2 * entries - (entries + 4 * math.ceil(entries / group_size)))
        value_counts.append(entries)

    benefits = [0.0] * len(descriptors)
    for (diagonal_block, rows, columns), members in shapes.items():
        if diagonal_block:
            local_rows, local_columns = torch.triu_indices(
                rows, columns, offset=1, device=device
            )
        else:
            grid_rows, grid_columns = torch.meshgrid(
                torch.arange(rows, device=device),
                torch.arange(columns, device=device),
                indexing="ij",
            )
            local_rows, local_columns = grid_rows.reshape(-1), grid_columns.reshape(-1)
        entries = int(local_rows.numel())
        if entries == 0:
            continue
        groups = math.ceil(entries / group_size)
        for start in range(0, len(members), batch_blocks):
            chunk = members[start : start + batch_blocks]
            row_starts = torch.tensor(
                [descriptors[index][0] * block_size for index in chunk], device=device
            )
            column_starts = torch.tensor(
                [descriptors[index][1] * block_size for index in chunk], device=device
            )
            source = matrix[
                row_starts[:, None] + local_rows[None, :],
                column_starts[:, None] + local_columns[None, :],
            ]
            padded = torch.zeros(
                (len(chunk), groups * group_size), device=device, dtype=source.dtype
            )
            padded[:, :entries] = source
            grouped = padded.reshape(len(chunk), groups, group_size)
            maxima = grouped.abs().amax(2)
            scales = torch.where(maxima > 0, maxima / 127.0, torch.ones_like(maxima)).to(
                torch.float32
            )
            quantized = torch.round(grouped / scales.to(grouped.dtype)[:, :, None]).clamp(
                -127, 127
            ).to(torch.int8)
            decoded = quantized.reshape(len(chunk), -1)[:, :entries].to(source.dtype) * scales.to(
                source.dtype
            ).repeat_interleave(group_size, dim=1)[:, :entries]
            int8_error = (decoded - source).square().sum(1)
            fp16_error = (source.to(torch.float16).to(source.dtype) - source).square().sum(1)
            values = (int8_error - fp16_error).to(torch.float64).cpu().tolist()
            for index, value in zip(chunk, values):
                benefits[index] = value
    return benefits, extra_costs, value_counts


class CriterionAdaptiveSquareRootBackend(SquareRootBackend):
    """Adaptive square-root backend whose block selection follows a declared criterion.

    ``update`` repeats :meth:`SquareRootBackend.update` statement by statement;
    only the compression call is replaced.  An optional read-only hook receives
    the unquantized factor before compression, for benchmarking.
    """

    def __init__(
        self,
        *,
        selection_criterion: str,
        adaptive_budget_fraction: float,
        pre_compression_hook: Callable[[torch.Tensor], None] | None = None,
        **kwargs,
    ) -> None:
        if selection_criterion not in CRITERIA:
            raise ValueError(f"unknown selection criterion {selection_criterion!r}")
        if "storage_mode" in kwargs:
            raise ValueError("criterion backends always use adaptive_int8_fp16 storage")
        super().__init__(
            storage_mode="adaptive_int8_fp16",
            adaptive_budget_fraction=adaptive_budget_fraction,
            **kwargs,
        )
        self.selection_criterion = selection_criterion
        self._pre_compression_hook = pre_compression_hook
        self.diagnostics.update(selection_criterion=selection_criterion)

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

        if self._pre_compression_hook is not None:
            self._pre_compression_hook(exact_upper)
        row_weights = (
            factor_row_weights(exact_upper)
            if self.selection_criterion == "row_weighted_system"
            else None
        )
        compressed, relative_factor_error, adaptive_diagnostics = (
            compress_upper_with_criterion(
                exact_upper,
                block_size=self.block_size,
                group_size=self.group_size,
                budget_fraction=float(self.adaptive_budget_fraction),
                criterion=self.selection_criterion,
                row_weights=row_weights,
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
            relative_local_factor_error=relative_factor_error,
            total_rows=self.total_rows,
        )
        self.diagnostics.update(adaptive_diagnostics)
        self.assert_exemplar_free_state()


__all__ = [
    "CRITERIA",
    "CriterionAdaptiveSquareRootBackend",
    "batched_factor_mse_benefits",
    "budget_bytes",
    "compress_upper_with_criterion",
    "factor_row_weights",
    "greedy_select",
    "score_blocks",
    "upper_block_descriptors",
]
