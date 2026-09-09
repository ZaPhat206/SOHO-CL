"""Correctness and accounting gates for M11 adaptive precision."""

from __future__ import annotations

import torch

from methods.analytic_ridge import (
    AdaptiveCompressedUpper,
    CompressedUpper,
    SquareRootBackend,
    persistent_tensor_bytes,
)


def _upper(dimension: int = 29) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1101)
    values = torch.randn(dimension, dimension, generator=generator)
    system = values.T @ values + 7.0 * torch.eye(dimension)
    return torch.linalg.cholesky(system).T


def test_adaptive_factor_obeys_budget_and_improves_int8_reconstruction():
    exact = _upper()
    int8_work = exact.clone()
    # Exercise the same concrete encoder used by the fixed P2B backend.
    int8_factor, int8_error = CompressedUpper.from_upper_inplace_streaming(
        int8_work,
        block_size=8,
        group_size=5,
        mode="int8",
    )
    adaptive_work = exact.clone()
    adaptive, adaptive_error, diagnostics = AdaptiveCompressedUpper.from_upper_inplace(
        adaptive_work,
        block_size=8,
        group_size=5,
        budget_fraction=0.25,
    )

    assert adaptive_error <= int8_error
    assert diagnostics["selected_fp16_blocks"] > 0
    assert diagnostics["used_extra_bytes"] <= diagnostics["extra_budget_bytes"]
    assert adaptive.persistent_state_bytes() == persistent_tensor_bytes(
        adaptive.persistent_tensors("factor")
    )
    assert adaptive.persistent_state_bytes() > persistent_tensor_bytes(
        int8_factor.persistent_tensors("factor")
    )
    torch.testing.assert_close(
        adaptive.reconstruct_upper(), adaptive_work, rtol=0, atol=0
    )


def test_adaptive_backend_checkpoint_round_trip_and_counts_mask():
    kwargs = dict(
        dimension=24,
        ridge_lambda=5.0,
        storage_mode="adaptive_int8_fp16",
        adaptive_budget_fraction=0.25,
        block_size=6,
        group_size=5,
        update_panel_size=7,
        quantization_backend="streaming",
        quantization_batch_blocks=3,
    )
    first = SquareRootBackend(**kwargs)
    generator = torch.Generator().manual_seed(1102)
    codes = torch.randn(31, 24, generator=generator)
    labels = torch.arange(5).repeat(7)[:31]
    first.update(codes, labels)
    assert "factor.precision_mask" in first.persistent_tensors()
    assert first.diagnostics["selected_fp16_blocks"] > 0

    restored = SquareRootBackend(**kwargs)
    restored.load_state_dict(first.state_dict())
    torch.testing.assert_close(restored.weights, first.weights, rtol=0, atol=0)
    assert restored.persistent_state_bytes() == first.persistent_state_bytes()
    assert torch.equal(
        restored.factor.precision_mask, first.factor.precision_mask
    )


def test_adaptive_selection_is_deterministic_and_label_free():
    first_work = _upper(25)
    second_work = first_work.clone()
    first, first_error, first_diagnostics = AdaptiveCompressedUpper.from_upper_inplace(
        first_work, block_size=7, group_size=6, budget_fraction=0.4
    )
    second, second_error, second_diagnostics = AdaptiveCompressedUpper.from_upper_inplace(
        second_work, block_size=7, group_size=6, budget_fraction=0.4
    )
    assert first_error == second_error
    assert first_diagnostics == second_diagnostics
    assert torch.equal(first.precision_mask, second.precision_mask)
    for left, right in zip(first.blocks, second.blocks):
        assert torch.equal(left.values, right.values)
        if left.scales is not None:
            assert torch.equal(left.scales, right.scales)
