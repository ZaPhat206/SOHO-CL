"""Correctness and accounting gates for same-byte scale-refined INT8."""

from __future__ import annotations

import torch

from methods.analytic_ridge.backends import SquareRootBackend
from methods.analytic_ridge.refined_backend import ScaleRefinedSquareRootBackend
from methods.analytic_ridge.refined_upper import (
    _refined_groupwise_int8_rows,
    compress_upper_refined_inplace_streaming,
)


def _backend(cls, dimension: int):
    common = dict(
        dimension=dimension,
        ridge_lambda=10.0,
        device="cpu",
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        block_size=5,
        group_size=4,
        update_panel_size=4,
        update_trailing_chunk_size=None,
        first_update_backend="gram_cholesky",
        quantization_batch_blocks=3,
    )
    if cls is SquareRootBackend:
        return cls(storage_mode="int8", quantization_backend="streaming", **common)
    return cls(scale_refinement_iterations=4, **common)


def test_refined_groups_never_exceed_maxabs_error_and_are_deterministic():
    values = torch.tensor(
        [
            [19.0, 0.12, -0.08, 0.03, 2.5, -2.2, 2.1],
            [0.0, 0.0, 0.0, 0.0, 3.0, 0.05, -0.04],
        ],
        dtype=torch.float32,
    )
    first = _refined_groupwise_int8_rows(values, 4, 4)
    second = _refined_groupwise_int8_rows(values, 4, 4)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert float(first[2]) <= float(first[3]) + 1e-12
    assert bool(torch.isfinite(first[1]).all())
    assert bool((first[1] > 0).all())


def test_refined_upper_has_same_payload_shape_and_no_larger_local_error():
    generator = torch.Generator().manual_seed(1101)
    source = torch.triu(torch.randn(13, 13, generator=generator))
    source.diagonal().abs_().add_(2.0)
    work = source.clone()
    compressed, refined_error, maxabs_error = (
        compress_upper_refined_inplace_streaming(
            work,
            block_size=5,
            group_size=4,
            maximum_batched_blocks=3,
            refinement_iterations=4,
        )
    )
    assert refined_error <= maxabs_error + 1e-12
    assert torch.equal(work, compressed.reconstruct_upper())
    assert bool((work.diagonal() > 0).all())


def test_refined_backend_matches_p2b_state_bytes_and_round_trips():
    generator = torch.Generator().manual_seed(1102)
    features = torch.randn(42, 11, generator=generator)
    labels = torch.arange(6).repeat_interleave(7)
    baseline = _backend(SquareRootBackend, 11)
    refined = _backend(ScaleRefinedSquareRootBackend, 11)
    baseline.update(features[:24], labels[:24])
    refined.update(features[:24], labels[:24])
    assert refined.persistent_state_bytes() == baseline.persistent_state_bytes()
    assert refined.diagnostics["relative_local_factor_error"] <= refined.diagnostics[
        "maxabs_same_input_relative_factor_error"
    ] + 1e-12

    restored = _backend(ScaleRefinedSquareRootBackend, 11)
    restored.load_state_dict(refined.state_dict())
    probe = torch.randn(9, 11, generator=generator)
    assert torch.equal(refined.predict(probe), restored.predict(probe))
    assert torch.allclose(
        refined.predict_logits(probe), restored.predict_logits(probe), atol=0, rtol=0
    )
    baseline.update(features[24:], labels[24:])
    refined.update(features[24:], labels[24:])
    restored.update(features[24:], labels[24:])
    assert torch.equal(refined.predict(probe), restored.predict(probe))
    assert refined.persistent_state_bytes() == baseline.persistent_state_bytes()
