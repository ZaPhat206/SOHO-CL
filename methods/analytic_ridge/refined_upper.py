"""Same-byte INT8 upper-factor storage with deterministic scale refinement."""

from __future__ import annotations

from collections import defaultdict
import math

import torch

from methods.srq_fly_optimized.storage import CompressedUpper, UpperBlock


def _refined_groupwise_int8_rows(
    values: torch.Tensor,
    group_size: int,
    refinement_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize rows while monotonically refining each group's LS scale.

    The max-absolute encoding is retained as the initial candidate.  Each
    iteration alternates the least-squares scale for the current integer
    assignments with nearest-code reassignment.  A per-group acceptance mask
    makes the non-increasing squared-error contract explicit despite finite
    precision.  No additional tensor is persistent: the returned values and
    scales have exactly the version-1 INT8 checkpoint shapes.
    """
    if values.ndim != 2 or group_size <= 0 or refinement_iterations <= 0:
        raise ValueError("invalid scale-refinement inputs")
    rows, columns = values.shape
    if columns == 0:
        empty_values = torch.empty(
            (rows, 0), device=values.device, dtype=torch.int8
        )
        empty_scales = torch.empty(
            (rows, 0), device=values.device, dtype=torch.float32
        )
        zero = torch.zeros((), device=values.device, dtype=torch.float64)
        return empty_values, empty_scales, zero, zero

    groups = math.ceil(columns / group_size)
    padded = torch.zeros(
        (rows, groups * group_size), device=values.device, dtype=values.dtype
    )
    padded[:, :columns] = values
    grouped = padded.reshape(rows, groups, group_size)
    maxima = grouped.abs().amax(2)
    scales = torch.where(
        maxima > 0, maxima / 127.0, torch.ones_like(maxima)
    ).to(torch.float32)
    assignments = torch.round(
        grouped / scales.to(grouped.dtype)[:, :, None]
    ).clamp(-127, 127).to(torch.int8)
    decoded = assignments.to(grouped.dtype) * scales.to(
        grouped.dtype
    )[:, :, None]
    baseline_error = (decoded - grouped).square().sum(dtype=torch.float64)
    group_error = (decoded - grouped).square().sum(2)

    for _ in range(refinement_iterations):
        work_assignments = assignments.to(grouped.dtype)
        denominator = work_assignments.square().sum(2)
        ls_scale = torch.where(
            denominator > 0,
            (grouped * work_assignments).sum(2) / denominator.clamp_min(1.0),
            scales.to(grouped.dtype),
        ).to(torch.float32)
        candidate = torch.round(
            grouped / ls_scale.to(grouped.dtype)[:, :, None]
        ).clamp(-127, 127).to(torch.int8)
        candidate_decoded = candidate.to(grouped.dtype) * ls_scale.to(
            grouped.dtype
        )[:, :, None]
        candidate_error = (candidate_decoded - grouped).square().sum(2)
        accept = candidate_error <= group_error
        assignments = torch.where(accept[:, :, None], candidate, assignments)
        scales = torch.where(accept, ls_scale, scales)
        group_error = torch.where(accept, candidate_error, group_error)

    refined_decoded = assignments.to(grouped.dtype) * scales.to(
        grouped.dtype
    )[:, :, None]
    refined_error = (refined_decoded - grouped).square().sum(dtype=torch.float64)
    return (
        assignments.reshape(rows, -1)[:, :columns],
        scales,
        refined_error,
        baseline_error,
    )


def compress_upper_refined_inplace_streaming(
    matrix: torch.Tensor,
    *,
    block_size: int,
    group_size: int,
    maximum_batched_blocks: int,
    refinement_iterations: int,
) -> tuple[CompressedUpper, float, float]:
    """Compress an upper factor with bounded workspace and refined scales."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("matrix must be non-empty and square")
    if matrix.dtype not in {torch.float32, torch.float64}:
        raise ValueError("matrix must use float32 or float64")
    if min(
        block_size,
        group_size,
        maximum_batched_blocks,
        refinement_iterations,
    ) <= 0:
        raise ValueError("storage and refinement sizes must be positive")

    finite = torch.ones((), device=matrix.device, dtype=torch.bool)
    validation_rows = max(block_size, min(len(matrix), 1024))
    for start in range(0, len(matrix), validation_rows):
        finite.logical_and_(
            torch.isfinite(matrix[start : start + validation_rows]).all()
        )
    if not bool(finite):
        raise ValueError("matrix contains NaN or Inf")

    diagonal = matrix.diagonal().to(torch.float32).clone()
    count = math.ceil(len(matrix) / block_size)
    descriptors: list[tuple[int, int, int]] = []
    by_length: dict[int, list[int]] = defaultdict(list)
    for row_block in range(count):
        rs, re = row_block * block_size, min(
            (row_block + 1) * block_size, len(matrix)
        )
        rows = re - rs
        for col_block in range(row_block, count):
            cs, ce = col_block * block_size, min(
                (col_block + 1) * block_size, len(matrix)
            )
            columns = ce - cs
            length = (
                rows * columns
                if row_block != col_block
                else rows * (rows - 1) // 2
            )
            index = len(descriptors)
            descriptors.append((row_block, col_block, length))
            by_length[length].append(index)

    encoded_blocks: list[tuple[torch.Tensor, torch.Tensor] | None] = [
        None
    ] * len(descriptors)
    original_diagonal = matrix.diagonal().to(torch.float64)
    squared_reference = original_diagonal.square().sum()
    refined_squared_error = (
        diagonal.to(torch.float64) - original_diagonal
    ).square().sum()
    maxabs_squared_error = refined_squared_error.clone()
    matrix.diagonal().copy_(diagonal.to(matrix.dtype))

    for indices in by_length.values():
        for start in range(0, len(indices), maximum_batched_blocks):
            chunk = indices[start : start + maximum_batched_blocks]
            source_rows = []
            for index in chunk:
                row_block, col_block, _ = descriptors[index]
                rs, re = row_block * block_size, min(
                    (row_block + 1) * block_size, len(matrix)
                )
                cs, ce = col_block * block_size, min(
                    (col_block + 1) * block_size, len(matrix)
                )
                local = matrix[rs:re, cs:ce]
                if row_block == col_block:
                    local_indices = torch.triu_indices(
                        len(local), len(local), offset=1, device=matrix.device
                    )
                    values = local[local_indices[0], local_indices[1]]
                else:
                    values = local.contiguous().view(-1)
                source_rows.append(values)
            stacked = torch.stack(source_rows)
            del source_rows

            encoded, scales, refined_error, maxabs_error = (
                _refined_groupwise_int8_rows(
                    stacked, group_size, refinement_iterations
                )
            )
            decoded = encoded.to(stacked.dtype) * scales.to(
                stacked.dtype
            ).repeat_interleave(group_size, dim=1)[:, : stacked.shape[1]]
            refined_squared_error.add_(refined_error)
            maxabs_squared_error.add_(maxabs_error)
            squared_reference.add_(stacked.square().sum(dtype=torch.float64))
            for row, index in enumerate(chunk):
                encoded_blocks[index] = (encoded[row], scales[row])
                row_block, col_block, _ = descriptors[index]
                rs, re = row_block * block_size, min(
                    (row_block + 1) * block_size, len(matrix)
                )
                cs, ce = col_block * block_size, min(
                    (col_block + 1) * block_size, len(matrix)
                )
                local = matrix[rs:re, cs:ce]
                if row_block == col_block:
                    local_indices = torch.triu_indices(
                        len(local), len(local), offset=1, device=matrix.device
                    )
                    local[local_indices[0], local_indices[1]] = decoded[row]
                else:
                    local.copy_(decoded[row].reshape(re - rs, ce - cs))
            del stacked, decoded

    blocks = []
    for descriptor, encoded in zip(descriptors, encoded_blocks):
        if encoded is None:
            raise RuntimeError("internal refined-block packing failure")
        row_block, col_block, _ = descriptor
        blocks.append(UpperBlock(row_block, col_block, encoded[0], encoded[1]))
    compressed = CompressedUpper(
        dimension=len(matrix),
        block_size=block_size,
        group_size=group_size,
        mode="int8",
        diagonal=diagonal,
        blocks=blocks,
        validate_values=False,
    )
    denominator = torch.clamp(squared_reference, min=1.0)
    refined_relative_error = float(
        torch.sqrt(refined_squared_error / denominator).item()
    )
    maxabs_relative_error = float(
        torch.sqrt(maxabs_squared_error / denominator).item()
    )
    return compressed, refined_relative_error, maxabs_relative_error


__all__ = [
    "_refined_groupwise_int8_rows",
    "compress_upper_refined_inplace_streaming",
]
