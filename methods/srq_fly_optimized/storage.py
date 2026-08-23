"""Full-coordinate upper-triangular storage for SRQ-FLY diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections import defaultdict

import torch


@dataclass(frozen=True)
class UpperBlock:
    row_block: int
    col_block: int
    values: torch.Tensor
    scales: torch.Tensor | None


def _groupwise_int8(values: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    flat = values.reshape(-1)
    if not len(flat):
        return (
            torch.empty(0, device=flat.device, dtype=torch.int8),
            torch.empty(0, device=flat.device, dtype=torch.float32),
        )
    groups = math.ceil(len(flat) / group_size)
    padded = torch.zeros(groups * group_size, device=flat.device, dtype=flat.dtype)
    padded[: len(flat)] = flat
    rows = padded.reshape(groups, group_size)
    maxima = rows.abs().amax(1)
    scales = torch.where(maxima > 0, maxima / 127.0, torch.ones_like(maxima)).to(
        torch.float32
    )
    quantized = torch.round(rows / scales.to(rows.dtype)[:, None]).clamp(
        -127, 127
    ).to(torch.int8)
    return quantized.reshape(-1)[: len(flat)], scales


def _groupwise_int8_rows(
    values: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized equivalent of ``_groupwise_int8`` for equal-length rows.

    Quantization groups restart at every row, exactly as they do for every
    ``UpperBlock`` in the checkpoint format.  Batching equal-sized blocks
    removes hundreds of small CUDA reductions without changing the stored
    values or scales.
    """
    if values.ndim != 2:
        raise ValueError("values must be a matrix")
    rows, columns = values.shape
    if columns == 0:
        return (
            torch.empty((rows, 0), device=values.device, dtype=torch.int8),
            torch.empty((rows, 0), device=values.device, dtype=torch.float32),
        )
    groups = math.ceil(columns / group_size)
    padded = torch.zeros(
        (rows, groups * group_size), device=values.device, dtype=values.dtype
    )
    padded[:, :columns] = values
    grouped = padded.reshape(rows, groups, group_size)
    maxima = grouped.abs().amax(2)
    scales = torch.where(maxima > 0, maxima / 127.0, torch.ones_like(maxima)).to(
        torch.float32
    )
    quantized = torch.round(grouped / scales.to(grouped.dtype)[:, :, None]).clamp(
        -127, 127
    ).to(torch.int8)
    return quantized.reshape(rows, -1)[:, :columns], scales


def _decode_groupwise(
    values: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    expanded = scales.to(dtype).repeat_interleave(group_size)[: len(values)]
    return values.to(dtype) * expanded


class CompressedUpper:
    """Exact diagonal plus compressed strict-upper entries.

    ``mode='int8'`` uses deterministic groupwise symmetric quantization.
    ``mode='float16'`` stores strict-upper values directly in float16.
    """

    VERSION = 1
    MODES = {"int8", "float16"}

    def __init__(
        self,
        *,
        dimension: int,
        block_size: int,
        group_size: int,
        mode: str,
        diagonal: torch.Tensor,
        blocks: list[UpperBlock],
        validate_values: bool = True,
    ) -> None:
        if dimension <= 0 or block_size <= 0 or group_size <= 0:
            raise ValueError("storage dimensions must be positive")
        if mode not in self.MODES:
            raise ValueError("mode must be int8 or float16")
        if diagonal.shape != (dimension,) or diagonal.dtype != torch.float32:
            raise ValueError("diagonal must be a float32 vector")
        if not bool(torch.isfinite(diagonal).all()):
            raise ValueError("diagonal contains NaN or Inf")
        self.dimension = int(dimension)
        self.block_size = int(block_size)
        self.group_size = int(group_size)
        self.mode = mode
        self.diagonal = diagonal
        self.blocks = blocks
        self._validate(validate_values=validate_values)

    @property
    def device(self) -> torch.device:
        return self.diagonal.device

    @property
    def block_count(self) -> int:
        return math.ceil(self.dimension / self.block_size)

    def _bounds(self, block: int) -> tuple[int, int]:
        start = block * self.block_size
        return start, min(start + self.block_size, self.dimension)

    def _validate(self, *, validate_values: bool) -> None:
        expected_count = self.block_count * (self.block_count + 1) // 2
        if len(self.blocks) != expected_count:
            raise ValueError("compressed upper triangle is incomplete")
        locations = set()
        for block in self.blocks:
            location = (block.row_block, block.col_block)
            if (
                block.row_block < 0
                or block.col_block < block.row_block
                or block.col_block >= self.block_count
                or location in locations
            ):
                raise ValueError("invalid or duplicate upper block")
            locations.add(location)
            rs, re = self._bounds(block.row_block)
            cs, ce = self._bounds(block.col_block)
            rows, columns = re - rs, ce - cs
            expected = rows * columns if block.row_block != block.col_block else rows * (rows - 1) // 2
            if block.values.numel() != expected or block.values.device != self.device:
                raise ValueError("compressed block shape/device mismatch")
            if self.mode == "int8":
                scale_count = math.ceil(expected / self.group_size)
                if (
                    block.values.dtype != torch.int8
                    or block.scales is None
                    or block.scales.shape != (scale_count,)
                    or block.scales.dtype != torch.float32
                    or block.scales.device != self.device
                ):
                    raise ValueError("invalid int8 block storage")
            elif (
                block.values.dtype != torch.float16
                or block.scales is not None
            ):
                raise ValueError("invalid float16 block storage")
        if not validate_values:
            return
        if self.mode == "int8":
            scales = [block.scales for block in self.blocks if block.scales is not None]
            if scales:
                packed_scales = torch.cat(scales)
                if not bool(torch.isfinite(packed_scales).all()) or bool(
                    (packed_scales <= 0).any()
                ):
                    raise ValueError("invalid int8 block storage")
        else:
            values = [block.values for block in self.blocks]
            if values and not bool(torch.isfinite(torch.cat(values)).all()):
                raise ValueError("invalid float16 block storage")

    @classmethod
    def from_upper(
        cls,
        matrix: torch.Tensor,
        *,
        block_size: int,
        group_size: int,
        mode: str,
    ) -> "CompressedUpper":
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
            raise ValueError("matrix must be non-empty and square")
        if matrix.dtype not in {torch.float32, torch.float64}:
            raise ValueError("matrix must use float32 or float64")
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("matrix contains NaN or Inf")
        diagonal = matrix.diagonal().to(torch.float32).clone()
        descriptors = []
        count = math.ceil(len(matrix) / block_size)
        for row_block in range(count):
            rs, re = row_block * block_size, min((row_block + 1) * block_size, len(matrix))
            for col_block in range(row_block, count):
                cs, ce = col_block * block_size, min((col_block + 1) * block_size, len(matrix))
                local = matrix[rs:re, cs:ce]
                if row_block == col_block:
                    indices = torch.triu_indices(len(local), len(local), offset=1, device=matrix.device)
                    values = local[indices[0], indices[1]]
                else:
                    values = local.reshape(-1)
                descriptors.append((row_block, col_block, values))
        if mode not in cls.MODES:
            raise ValueError("mode must be int8 or float16")

        # Full off-diagonal blocks share a shape, as do full diagonal blocks.
        # Quantize each equal-length family in one operation while preserving
        # the exact per-block grouping used by the version-1 checkpoint.
        by_length: dict[int, list[int]] = defaultdict(list)
        for index, (_, _, values) in enumerate(descriptors):
            by_length[values.numel()].append(index)
        encoded_blocks: list[tuple[torch.Tensor, torch.Tensor | None] | None] = [
            None
        ] * len(descriptors)
        maximum_batched_blocks = 16
        for indices in by_length.values():
            for start in range(0, len(indices), maximum_batched_blocks):
                chunk = indices[start : start + maximum_batched_blocks]
                stacked = torch.stack([descriptors[index][2] for index in chunk])
                if mode == "int8":
                    encoded, scales = _groupwise_int8_rows(stacked, group_size)
                    for row, index in enumerate(chunk):
                        encoded_blocks[index] = (encoded[row], scales[row])
                else:
                    encoded = stacked.to(torch.float16)
                    if not bool(torch.isfinite(encoded).all()):
                        raise ValueError("float16 compression overflowed to NaN or Inf")
                    for row, index in enumerate(chunk):
                        encoded_blocks[index] = (encoded[row], None)

        blocks = []
        for descriptor, encoded in zip(descriptors, encoded_blocks):
            if encoded is None:
                raise RuntimeError("internal compressed-block packing failure")
            row_block, col_block, _ = descriptor
            blocks.append(UpperBlock(row_block, col_block, encoded[0], encoded[1]))
        return cls(
            dimension=len(matrix),
            block_size=block_size,
            group_size=group_size,
            mode=mode,
            diagonal=diagonal,
            blocks=blocks,
            # ``matrix`` was checked once above and the encoders are
            # deterministic finite transforms.  Avoid one GPU synchronization
            # per block in this hot path; checkpoint loads remain strict.
            validate_values=False,
        )

    def reconstruct_upper(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("reconstruction dtype must be float32 or float64")
        matrix = torch.zeros(
            (self.dimension, self.dimension), device=self.device, dtype=dtype
        )
        matrix.diagonal().copy_(self.diagonal.to(dtype))
        for block in self.blocks:
            rs, re = self._bounds(block.row_block)
            cs, ce = self._bounds(block.col_block)
            if self.mode == "int8":
                decoded = _decode_groupwise(
                    block.values, block.scales, self.group_size, dtype
                )
            else:
                decoded = block.values.to(dtype)
            if block.row_block == block.col_block:
                size = re - rs
                indices = torch.triu_indices(size, size, offset=1, device=self.device)
                local = matrix[rs:re, cs:ce]
                local[indices[0], indices[1]] = decoded
            else:
                matrix[rs:re, cs:ce] = decoded.reshape(re - rs, ce - cs)
        return matrix

    def reconstruct_symmetric(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        upper = self.reconstruct_upper(dtype=dtype)
        return upper + upper.T - torch.diag_embed(upper.diagonal())

    def persistent_tensors(self, prefix: str) -> dict[str, torch.Tensor]:
        tensors = {f"{prefix}.diagonal": self.diagonal}
        for index, block in enumerate(self.blocks):
            tensors[f"{prefix}.block_{index}.values"] = block.values
            if block.scales is not None:
                tensors[f"{prefix}.block_{index}.scales"] = block.scales
        return tensors

    def state_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "dimension": self.dimension,
            "block_size": self.block_size,
            "group_size": self.group_size,
            "mode": self.mode,
            "diagonal": self.diagonal.detach().cpu().clone(),
            "blocks": [
                {
                    "row_block": block.row_block,
                    "col_block": block.col_block,
                    "values": block.values.detach().cpu().clone(),
                    "scales": None
                    if block.scales is None
                    else block.scales.detach().cpu().clone(),
                }
                for block in self.blocks
            ],
        }

    @classmethod
    def load_state_dict(cls, state: dict, *, device: str | torch.device) -> "CompressedUpper":
        if state.get("version") != cls.VERSION:
            raise ValueError("unsupported compressed-upper checkpoint")
        target = torch.device(device)
        blocks = [
            UpperBlock(
                row_block=int(item["row_block"]),
                col_block=int(item["col_block"]),
                values=item["values"].to(target),
                scales=None if item["scales"] is None else item["scales"].to(target),
            )
            for item in state["blocks"]
        ]
        return cls(
            dimension=int(state["dimension"]),
            block_size=int(state["block_size"]),
            group_size=int(state["group_size"]),
            mode=str(state["mode"]),
            diagonal=state["diagonal"].to(target),
            blocks=blocks,
            validate_values=True,
        )


def projected_srq_state_bytes(
    *,
    feature_dim: int,
    expand_dim: int,
    synaptic_degree: int,
    num_classes: int,
    block_size: int,
    group_size: int,
) -> dict[str, int | float]:
    """Tensor-byte projection under the repository's matched-state policy."""
    if min(
        feature_dim,
        expand_dim,
        synaptic_degree,
        num_classes,
        block_size,
        group_size,
    ) <= 0:
        raise ValueError("state dimensions must be positive")
    nonzeros = expand_dim * synaptic_degree
    projection = nonzeros * 4 + nonzeros * 8 + (feature_dim + 1) * 8
    diagonal = expand_dim * 4
    values = expand_dim * (expand_dim - 1) // 2
    scales = 0
    blocks = math.ceil(expand_dim / block_size)
    for row in range(blocks):
        rows = min(block_size, expand_dim - row * block_size)
        for column in range(row, blocks):
            columns = min(block_size, expand_dim - column * block_size)
            entries = rows * columns if row != column else rows * (rows - 1) // 2
            scales += math.ceil(entries / group_size) * 4
    cross_classifier_counts = 2 * expand_dim * num_classes * 4 + num_classes * 4
    compressed = projection + diagonal + values + scales + cross_classifier_counts
    exact = projection + expand_dim * expand_dim * 4 + cross_classifier_counts
    return {
        "projection_bytes": projection,
        "factor_bytes": diagonal + values + scales,
        "cross_classifier_count_bytes": cross_classifier_counts,
        "compressed_total_bytes": compressed,
        "exact_fly_total_bytes": exact,
        "state_fraction": compressed / exact,
    }

