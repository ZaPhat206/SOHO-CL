"""Random and task-1-static controls for SRQ mixed-precision storage.

These controls deliberately leave the locked SRQ-Adaptive implementation
unchanged.  They share its block layout, INT8/FP16 encodings, precision-mask
storage and byte-budget definition, but replace only the rule that chooses
which strict-upper blocks use FP16.

``random``
    Visits blocks in a deterministic pseudorandom order and accepts every
    block that still fits the byte ceiling.  The seed depends only on the
    declared allocation seed and the number of rows already processed; it
    never uses factor values, labels, logits or accuracy.

``static``
    Runs the locked factor-MSE selector at the first update, then reuses the
    same precision mask at every later update.  Values in all blocks continue
    to be updated and re-encoded normally.
"""

from __future__ import annotations

import math
import random

import torch

from .adaptive_selection import budget_bytes, greedy_select, score_blocks
from .adaptive_upper import AdaptiveCompressedUpper, AdaptiveUpperBlock, _groupwise_int8
from .backends import SquareRootBackend, _relative_factor_residual
from .qr import blocked_qr_rank_update


POLICIES = ("random", "static")


def _bounds(block: int, block_size: int, dimension: int) -> tuple[int, int]:
    start = block * block_size
    return start, min(start + block_size, dimension)


def block_layout(
    dimension: int, *, block_size: int, group_size: int
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return descriptors, value counts and FP16-over-INT8 costs."""
    if min(dimension, block_size, group_size) <= 0:
        raise ValueError("dimension, block_size and group_size must be positive")
    count = math.ceil(dimension / block_size)
    descriptors = []
    value_counts = []
    extra_costs = []
    for row_block in range(count):
        rs, re = _bounds(row_block, block_size, dimension)
        for col_block in range(row_block, count):
            cs, ce = _bounds(col_block, block_size, dimension)
            rows, columns = re - rs, ce - cs
            entries = rows * (rows - 1) // 2 if row_block == col_block else rows * columns
            descriptors.append((row_block, col_block))
            value_counts.append(entries)
            int8_bytes = entries + 4 * math.ceil(entries / group_size)
            extra_costs.append(2 * entries - int8_bytes)
    return descriptors, value_counts, extra_costs


def random_select(
    extra_costs: list[int], *, extra_budget: int, seed: int
) -> tuple[list[bool], int]:
    """Select blocks without replacement in a declared random order."""
    if extra_budget < 0:
        raise ValueError("extra_budget must be non-negative")
    order = list(range(len(extra_costs)))
    random.Random(int(seed)).shuffle(order)
    selected = [False] * len(extra_costs)
    used = 0
    for index in order:
        cost = int(extra_costs[index])
        if cost > 0 and used + cost <= extra_budget:
            selected[index] = True
            used += cost
    return selected, used


def compress_upper_with_mask(
    matrix: torch.Tensor,
    *,
    block_size: int,
    group_size: int,
    budget_fraction: float,
    selected: list[bool],
    selection_policy: str,
) -> tuple[AdaptiveCompressedUpper, float, dict[str, int | float | str]]:
    """Encode an upper factor using a supplied, budget-checked mask."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("matrix must be non-empty and square")
    if matrix.dtype not in {torch.float32, torch.float64}:
        raise ValueError("matrix must use float32 or float64")
    if not 0.0 <= float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must lie in [0, 1]")
    finite = torch.ones((), device=matrix.device, dtype=torch.bool)
    validation_rows = max(block_size, min(len(matrix), 1024))
    for start in range(0, len(matrix), validation_rows):
        finite.logical_and_(torch.isfinite(matrix[start : start + validation_rows]).all())
    if not bool(finite):
        raise ValueError("matrix contains NaN or Inf")

    dimension = len(matrix)
    descriptors, value_counts, extra_costs = block_layout(
        dimension, block_size=block_size, group_size=group_size
    )
    if len(selected) != len(descriptors):
        raise ValueError("precision mask does not match the upper-block layout")
    selected = [bool(value) for value in selected]
    base_bytes, full_bytes, extra_budget = budget_bytes(
        value_counts, group_size=group_size, budget_fraction=budget_fraction
    )
    used_extra = sum(cost for cost, flag in zip(extra_costs, selected) if flag)
    if any(flag and cost <= 0 for flag, cost in zip(selected, extra_costs)):
        raise ValueError("precision mask selected a block with non-positive FP16 cost")
    if used_extra > extra_budget:
        raise ValueError("precision mask exceeds the declared byte budget")

    diagonal = matrix.diagonal().to(torch.float32).clone()
    squared_error = torch.zeros((), device=matrix.device, dtype=torch.float64)
    squared_reference = diagonal.to(torch.float64).square().sum()
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
            local_indices = None
            source = local.contiguous().view(-1)
        squared_reference.add_(source.to(torch.float64).square().sum())
        if selected[index]:
            values = source.to(torch.float16)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("adaptive FP16 candidate overflowed")
            scales = None
            decoded = values.to(source.dtype)
        else:
            values, scales, decoded = _groupwise_int8(source, group_size)
        squared_error.add_((decoded.to(torch.float64) - source.to(torch.float64)).square().sum())
        blocks.append(AdaptiveUpperBlock(row_block, col_block, values, scales))
        if local_indices is None:
            local.copy_(decoded.reshape(re - rs, ce - cs))
        else:
            local[local_indices[0], local_indices[1]] = decoded

    matrix.diagonal().copy_(diagonal.to(matrix.dtype))
    relative_error = float(
        torch.sqrt(squared_error / torch.clamp(squared_reference, min=1.0)).item()
    )
    mask = torch.tensor(selected, device=matrix.device, dtype=torch.uint8)
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
            entries for entries, flag in zip(value_counts, selected) if flag
        ),
        "total_strict_upper_values": sum(value_counts),
        "int8_strict_upper_bytes": base_bytes,
        "fp16_strict_upper_bytes": full_bytes,
        "extra_budget_bytes": extra_budget,
        "used_extra_bytes": used_extra,
        "factor_persistent_bytes": state.persistent_state_bytes(),
        "factor_budget_ceiling_bytes": 4 * dimension + len(selected) + base_bytes + extra_budget,
        "selection_policy": selection_policy,
    }
    return state, relative_error, diagnostics


class SelectionControlSquareRootBackend(SquareRootBackend):
    """Square-root backend with random or task-1-static precision masks."""

    def __init__(
        self,
        *,
        selection_policy: str,
        adaptive_budget_fraction: float,
        allocation_seed: int = 2025,
        **kwargs,
    ) -> None:
        if selection_policy not in POLICIES:
            raise ValueError(f"unknown selection policy {selection_policy!r}")
        if "storage_mode" in kwargs:
            raise ValueError("selection controls always use adaptive_int8_fp16 storage")
        super().__init__(
            storage_mode="adaptive_int8_fp16",
            adaptive_budget_fraction=adaptive_budget_fraction,
            **kwargs,
        )
        self.selection_policy = selection_policy
        self.allocation_seed = int(allocation_seed)
        self.diagnostics.update(
            selection_policy=selection_policy,
            allocation_seed=self.allocation_seed,
        )

    def _choose_mask(self, factor: torch.Tensor) -> tuple[list[bool], str]:
        descriptors, counts, costs = block_layout(
            len(factor), block_size=self.block_size, group_size=self.group_size
        )
        _, _, extra_budget = budget_bytes(
            counts,
            group_size=self.group_size,
            budget_fraction=float(self.adaptive_budget_fraction),
        )
        if self.selection_policy == "static" and self.factor is not None:
            mask = self.factor.precision_mask.detach().cpu().tolist()
            if len(mask) != len(descriptors):
                raise RuntimeError("static precision mask changed shape")
            return [bool(value) for value in mask], "task1_factor_mse_mask"
        if self.selection_policy == "static":
            scores = score_blocks(
                factor,
                block_size=self.block_size,
                group_size=self.group_size,
                criterion="factor_mse",
            )
            selected, _ = greedy_select(scores["benefits"], scores["extra_costs"], extra_budget)
            return selected, "task1_factor_mse_selection"
        # total_rows is already persistent through counts and makes the next
        # mask reproducible after a checkpoint without storing an RNG state.
        seed = self.allocation_seed + 1_000_003 * int(self.total_rows)
        selected, _ = random_select(costs, extra_budget=extra_budget, seed=seed)
        return selected, "declared_random_seed_and_rows_seen"

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        values, target_labels = self._validated_batch(features, labels)
        class_ids, cross, counts, targets = self._expanded_statistics(target_labels)
        solve_values = values.to(self.solver_dtype)
        previous = None if self.factor is None else self.factor.reconstruct_upper(
            dtype=self.solver_dtype
        )
        if previous is None and self.first_update_backend == "implicit_ridge_qr":
            previous = torch.zeros(
                (self.dimension, self.dimension), device=self.device, dtype=self.solver_dtype
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

        selected, mask_source = self._choose_mask(exact_upper)
        compressed, relative_factor_error, control_diagnostics = compress_upper_with_mask(
            exact_upper,
            block_size=self.block_size,
            group_size=self.group_size,
            budget_fraction=float(self.adaptive_budget_fraction),
            selected=selected,
            selection_policy=self.selection_policy,
        )
        reconstructed = exact_upper
        if bool((reconstructed.diagonal() <= 0).any()):
            raise RuntimeError("compressed square-root diagonal is not positive")
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        work_cross = new_cross.to(self.solver_dtype)
        intermediate = torch.linalg.solve_triangular(reconstructed.T, work_cross, upper=False)
        weights = torch.linalg.solve_triangular(reconstructed, intermediate, upper=True)
        residual = _relative_factor_residual(reconstructed, weights, work_cross)

        self.factor = compressed
        self.class_ids, self.Q, self.counts = class_ids, new_cross, new_counts
        self.weights = weights
        self.total_rows += len(values)
        self.diagnostics.update(
            solver_relative_residual=residual,
            relative_local_factor_error=relative_factor_error,
            total_rows=self.total_rows,
            mask_source=mask_source,
        )
        self.diagnostics.update(control_diagnostics)
        self.assert_exemplar_free_state()


__all__ = [
    "POLICIES",
    "SelectionControlSquareRootBackend",
    "block_layout",
    "compress_upper_with_mask",
    "random_select",
]
