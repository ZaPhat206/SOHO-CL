"""Synthetic mathematical sanity tests for SRQ-FLY storage."""

import torch

from methods.srq_fly import CompressedUpper, projected_srq_state_bytes


def _spd(seed: int = 701, dimension: int = 19) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    return values.T @ values + 3.0 * torch.eye(dimension, dtype=torch.float64)


def test_groupwise_int8_is_deterministic_symmetric_and_preserves_diagonal():
    matrix = _spd()
    first = CompressedUpper.from_upper(matrix, block_size=6, group_size=7, mode="int8")
    second = CompressedUpper.from_upper(matrix, block_size=6, group_size=7, mode="int8")
    reconstructed = first.reconstruct_symmetric(dtype=torch.float64)
    torch.testing.assert_close(reconstructed, reconstructed.T, rtol=0, atol=0)
    torch.testing.assert_close(
        reconstructed.diagonal(), matrix.diagonal().to(torch.float32).to(torch.float64),
        rtol=0, atol=0,
    )
    for left, right in zip(first.blocks, second.blocks):
        assert torch.equal(left.values, right.values)
        assert torch.equal(left.scales, right.scales)


def test_square_root_quantization_is_spd_by_construction():
    exact_upper = torch.linalg.cholesky(_spd()).T
    for mode in ("float16", "int8"):
        state = CompressedUpper.from_upper(
            exact_upper, block_size=5, group_size=8, mode=mode
        )
        reconstructed = state.reconstruct_upper(dtype=torch.float64)
        assert bool((reconstructed.diagonal() > 0).all())
        eigenvalues = torch.linalg.eigvalsh(reconstructed.T @ reconstructed)
        assert float(eigenvalues.min()) > 0


def test_groupwise_quantization_improves_over_one_scale_for_heterogeneous_values():
    matrix = torch.zeros(12, 12, dtype=torch.float64)
    matrix.diagonal().fill_(10.0)
    upper = torch.triu_indices(12, 12, offset=1)
    values = torch.logspace(-4, 4, upper.shape[1], dtype=torch.float64)
    matrix[upper[0], upper[1]] = values
    grouped = CompressedUpper.from_upper(matrix, block_size=12, group_size=4, mode="int8")
    one_scale = CompressedUpper.from_upper(
        matrix, block_size=12, group_size=len(values), mode="int8"
    )
    grouped_error = torch.linalg.vector_norm(grouped.reconstruct_upper(dtype=torch.float64) - matrix)
    one_scale_error = torch.linalg.vector_norm(one_scale.reconstruct_upper(dtype=torch.float64) - matrix)
    assert grouped_error < one_scale_error


def test_projected_srq_state_is_below_quarter_of_exact_fly():
    projection = projected_srq_state_bytes(
        feature_dim=768,
        expand_dim=10_000,
        synaptic_degree=300,
        num_classes=200,
        block_size=256,
        group_size=64,
    )
    assert projection["compressed_total_bytes"] < projection["exact_fly_total_bytes"]
    assert projection["state_fraction"] <= 0.25
    assert 100_000_000 < projection["compressed_total_bytes"] < 110_000_000


def test_compressed_checkpoint_rejects_structure_mismatch():
    state = CompressedUpper.from_upper(
        _spd(dimension=9), block_size=4, group_size=3, mode="int8"
    ).state_dict()
    state["blocks"] = state["blocks"][:-1]
    try:
        CompressedUpper.load_state_dict(state, device="cpu")
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("an incomplete compressed triangle must fail")


def test_float16_storage_fails_on_overflow_instead_of_persisting_inf():
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[0, 1] = 1e10
    try:
        CompressedUpper.from_upper(matrix, block_size=4, group_size=2, mode="float16")
    except ValueError as error:
        assert "float16" in str(error)
    else:
        raise AssertionError("float16 overflow must fail closed")
