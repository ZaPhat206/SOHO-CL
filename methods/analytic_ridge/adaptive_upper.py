"""Budget-locked mixed INT8/FP16 storage for an upper-triangular factor."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AdaptiveUpperBlock:
    row_block: int
    col_block: int
    values: torch.Tensor
    scales: torch.Tensor | None


def _groupwise_int8(
    values: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = values.reshape(-1)
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
    encoded = quantized.reshape(-1)[: len(flat)]
    decoded = encoded.to(flat.dtype) * scales.to(flat.dtype).repeat_interleave(
        group_size
    )[: len(flat)]
    return encoded, scales, decoded


class AdaptiveCompressedUpper:
    """FP32 diagonal plus a byte-budgeted mixture of INT8 and FP16 blocks.

    Every strict-upper block is first evaluated under both encodings. Blocks
    are promoted to FP16 greedily by reduction in squared reconstruction error
    per additional byte. The byte ceiling is fixed by ``budget_fraction``
    between all-INT8 and all-FP16 strict-upper payloads. The precision mask is
    explicit persistent tensor state and is included in the ceiling.
    """

    VERSION = 1
    MODE = "adaptive_int8_fp16"

    def __init__(
        self,
        *,
        dimension: int,
        block_size: int,
        group_size: int,
        budget_fraction: float,
        diagonal: torch.Tensor,
        precision_mask: torch.Tensor,
        blocks: list[AdaptiveUpperBlock],
        validate_values: bool = True,
    ) -> None:
        if dimension <= 0 or block_size <= 0 or group_size <= 0:
            raise ValueError("storage dimensions must be positive")
        if not 0.0 <= float(budget_fraction) <= 1.0:
            raise ValueError("budget_fraction must lie in [0, 1]")
        if diagonal.shape != (dimension,) or diagonal.dtype != torch.float32:
            raise ValueError("diagonal must be a float32 vector")
        if not bool(torch.isfinite(diagonal).all()):
            raise ValueError("diagonal contains NaN or Inf")
        self.dimension = int(dimension)
        self.block_size = int(block_size)
        self.group_size = int(group_size)
        self.budget_fraction = float(budget_fraction)
        self.mode = self.MODE
        self.diagonal = diagonal
        self.precision_mask = precision_mask
        self.blocks = blocks
        self._validate(validate_values=validate_values)

    @property
    def device(self) -> torch.device:
        return self.diagonal.device

    @property
    def block_count(self) -> int:
        return math.ceil(self.dimension / self.block_size)

    @property
    def selected_fp16_blocks(self) -> int:
        return int(self.precision_mask.sum().item())

    def _bounds(self, block: int) -> tuple[int, int]:
        start = block * self.block_size
        return start, min(start + self.block_size, self.dimension)

    def _validate(self, *, validate_values: bool) -> None:
        expected_count = self.block_count * (self.block_count + 1) // 2
        if len(self.blocks) != expected_count:
            raise ValueError("adaptive upper triangle is incomplete")
        if (
            self.precision_mask.shape != (expected_count,)
            or self.precision_mask.dtype != torch.uint8
            or self.precision_mask.device != self.device
            or bool((self.precision_mask > 1).any())
        ):
            raise ValueError("invalid adaptive precision mask")
        mask_values = self.precision_mask.cpu().tolist()
        locations = set()
        for index, block in enumerate(self.blocks):
            location = (block.row_block, block.col_block)
            if (
                block.row_block < 0
                or block.col_block < block.row_block
                or block.col_block >= self.block_count
                or location in locations
            ):
                raise ValueError("invalid or duplicate adaptive upper block")
            locations.add(location)
            rs, re = self._bounds(block.row_block)
            cs, ce = self._bounds(block.col_block)
            rows, columns = re - rs, ce - cs
            expected = (
                rows * columns
                if block.row_block != block.col_block
                else rows * (rows - 1) // 2
            )
            if block.values.numel() != expected or block.values.device != self.device:
                raise ValueError("adaptive block shape/device mismatch")
            is_fp16 = bool(mask_values[index])
            if is_fp16:
                if block.values.dtype != torch.float16 or block.scales is not None:
                    raise ValueError("invalid adaptive FP16 block")
            else:
                scale_count = math.ceil(expected / self.group_size)
                if (
                    block.values.dtype != torch.int8
                    or block.scales is None
                    or block.scales.shape != (scale_count,)
                    or block.scales.dtype != torch.float32
                    or block.scales.device != self.device
                ):
                    raise ValueError("invalid adaptive INT8 block")
            if validate_values and is_fp16 and not bool(
                torch.isfinite(block.values).all()
            ):
                raise ValueError("adaptive FP16 block contains NaN or Inf")
            if validate_values and not is_fp16 and (
                not bool(torch.isfinite(block.scales).all())
                or bool((block.scales <= 0).any())
            ):
                raise ValueError("adaptive INT8 scales are invalid")

    @classmethod
    def from_upper_inplace(
        cls,
        matrix: torch.Tensor,
        *,
        block_size: int,
        group_size: int,
        budget_fraction: float,
    ) -> tuple["AdaptiveCompressedUpper", float, dict[str, int | float]]:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
            raise ValueError("matrix must be non-empty and square")
        if matrix.dtype not in {torch.float32, torch.float64}:
            raise ValueError("matrix must use float32 or float64")
        if not 0.0 <= float(budget_fraction) <= 1.0:
            raise ValueError("budget_fraction must lie in [0, 1]")
        if min(block_size, group_size) <= 0:
            raise ValueError("block_size and group_size must be positive")
        finite = torch.ones((), device=matrix.device, dtype=torch.bool)
        validation_rows = max(block_size, min(len(matrix), 1024))
        for start in range(0, len(matrix), validation_rows):
            finite.logical_and_(
                torch.isfinite(matrix[start : start + validation_rows]).all()
            )
        if not bool(finite):
            raise ValueError("matrix contains NaN or Inf")

        dimension = len(matrix)
        diagonal = matrix.diagonal().to(torch.float32).clone()
        count = math.ceil(dimension / block_size)
        descriptors: list[tuple[int, int]] = []
        int8_blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
        int8_errors: list[torch.Tensor] = []
        fp16_errors: list[torch.Tensor] = []
        reference_sums: list[torch.Tensor] = []
        extra_costs: list[int] = []
        value_counts: list[int] = []

        for row_block in range(count):
            rs, re = row_block * block_size, min(
                (row_block + 1) * block_size, dimension
            )
            for col_block in range(row_block, count):
                cs, ce = col_block * block_size, min(
                    (col_block + 1) * block_size, dimension
                )
                local = matrix[rs:re, cs:ce]
                if row_block == col_block:
                    indices = torch.triu_indices(
                        re - rs, ce - cs, offset=1, device=matrix.device
                    )
                    source = local[indices[0], indices[1]]
                else:
                    source = local.contiguous().view(-1)
                encoded, scales, decoded_int8 = _groupwise_int8(
                    source, group_size
                )
                encoded_fp16 = source.to(torch.float16)
                if not bool(torch.isfinite(encoded_fp16).all()):
                    raise ValueError("adaptive FP16 candidate overflowed")
                decoded_fp16 = encoded_fp16.to(source.dtype)
                descriptors.append((row_block, col_block))
                int8_blocks.append((encoded, scales))
                int8_errors.append((decoded_int8 - source).square().sum())
                fp16_errors.append((decoded_fp16 - source).square().sum())
                reference_sums.append(source.square().sum())
                entries = int(source.numel())
                int8_bytes = entries + 4 * math.ceil(entries / group_size)
                fp16_bytes = 2 * entries
                extra_costs.append(fp16_bytes - int8_bytes)
                value_counts.append(entries)

        benefit = torch.stack(int8_errors) - torch.stack(fp16_errors)
        benefit_cpu = benefit.detach().to(torch.float64).cpu().tolist()
        base_strict_bytes = sum(
            entries + 4 * math.ceil(entries / group_size)
            for entries in value_counts
        )
        full_fp16_strict_bytes = 2 * sum(value_counts)
        extra_budget = math.floor(
            float(budget_fraction)
            * (full_fp16_strict_bytes - base_strict_bytes)
        )
        order = sorted(
            range(len(descriptors)),
            key=lambda index: (
                -benefit_cpu[index] / max(extra_costs[index], 1),
                index,
            ),
        )
        selected = [False] * len(descriptors)
        used_extra = 0
        for index in order:
            cost = extra_costs[index]
            if benefit_cpu[index] <= 0 or cost <= 0:
                continue
            if used_extra + cost <= extra_budget:
                selected[index] = True
                used_extra += cost

        blocks: list[AdaptiveUpperBlock] = []
        for index, (row_block, col_block) in enumerate(descriptors):
            rs, re = row_block * block_size, min(
                (row_block + 1) * block_size, dimension
            )
            cs, ce = col_block * block_size, min(
                (col_block + 1) * block_size, dimension
            )
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
                values, scales = int8_blocks[index]
                decoded = values.to(source.dtype) * scales.to(
                    source.dtype
                ).repeat_interleave(group_size)[: len(values)]
            blocks.append(
                AdaptiveUpperBlock(row_block, col_block, values, scales)
            )
            if row_block == col_block:
                local[local_indices[0], local_indices[1]] = decoded
            else:
                local.copy_(decoded.reshape(re - rs, ce - cs))

        original_diagonal = matrix.diagonal().to(torch.float64)
        diagonal_error = (diagonal.to(torch.float64) - original_diagonal).square().sum()
        diagonal_reference = original_diagonal.square().sum()
        mask = torch.tensor(selected, device=matrix.device, dtype=torch.uint8)
        chosen_errors = torch.where(
            mask.to(torch.bool), torch.stack(fp16_errors), torch.stack(int8_errors)
        )
        squared_error = diagonal_error + chosen_errors.to(torch.float64).sum()
        squared_reference = diagonal_reference + torch.stack(reference_sums).to(
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
        state = cls(
            dimension=dimension,
            block_size=block_size,
            group_size=group_size,
            budget_fraction=budget_fraction,
            diagonal=diagonal,
            precision_mask=mask,
            blocks=blocks,
            validate_values=False,
        )
        diagnostics: dict[str, int | float] = {
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
        }
        return state, relative_error, diagnostics

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
            if block.scales is None:
                decoded = block.values.to(dtype)
            else:
                decoded = block.values.to(dtype) * block.scales.to(
                    dtype
                ).repeat_interleave(self.group_size)[: len(block.values)]
            if block.row_block == block.col_block:
                indices = torch.triu_indices(
                    re - rs, ce - cs, offset=1, device=self.device
                )
                local = matrix[rs:re, cs:ce]
                local[indices[0], indices[1]] = decoded
            else:
                matrix[rs:re, cs:ce] = decoded.reshape(re - rs, ce - cs)
        return matrix

    def persistent_tensors(self, prefix: str) -> dict[str, torch.Tensor]:
        tensors = {
            f"{prefix}.diagonal": self.diagonal,
            f"{prefix}.precision_mask": self.precision_mask,
        }
        for index, block in enumerate(self.blocks):
            tensors[f"{prefix}.block_{index}.values"] = block.values
            if block.scales is not None:
                tensors[f"{prefix}.block_{index}.scales"] = block.scales
        return tensors

    def persistent_state_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.persistent_tensors("factor").values()
        )

    def state_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "mode": self.MODE,
            "dimension": self.dimension,
            "block_size": self.block_size,
            "group_size": self.group_size,
            "budget_fraction": self.budget_fraction,
            "diagonal": self.diagonal.detach().cpu().clone(),
            "precision_mask": self.precision_mask.detach().cpu().clone(),
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
    def load_state_dict(
        cls, state: dict, *, device: str | torch.device
    ) -> "AdaptiveCompressedUpper":
        if state.get("version") != cls.VERSION or state.get("mode") != cls.MODE:
            raise ValueError("unsupported adaptive compressed-upper checkpoint")
        target = torch.device(device)
        blocks = [
            AdaptiveUpperBlock(
                row_block=int(item["row_block"]),
                col_block=int(item["col_block"]),
                values=item["values"].to(target),
                scales=None
                if item["scales"] is None
                else item["scales"].to(target),
            )
            for item in state["blocks"]
        ]
        return cls(
            dimension=int(state["dimension"]),
            block_size=int(state["block_size"]),
            group_size=int(state["group_size"]),
            budget_fraction=float(state["budget_fraction"]),
            diagonal=state["diagonal"].to(target),
            precision_mask=state["precision_mask"].to(target),
            blocks=blocks,
            validate_values=True,
        )


__all__ = ["AdaptiveCompressedUpper", "AdaptiveUpperBlock"]
