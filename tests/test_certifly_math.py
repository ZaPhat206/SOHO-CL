"""Synthetic mathematical sanity tests for CertiFLY."""

import torch

from methods.certifly import (
    QuantizedSymmetricGram,
    certified_argmax_mask,
    logit_error_bound,
    projected_all_int8_state_bytes,
    solve_certified_ridge,
)


DTYPE = torch.float64


def _problem(seed=301, samples=48, dimension=18, classes=5):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randn(samples, dimension, generator=generator, dtype=DTYPE)
    targets = torch.nn.functional.one_hot(
        torch.arange(samples) % classes, num_classes=classes
    ).to(DTYPE)
    return codes, codes.T @ codes, codes.T @ targets


def test_quantized_gram_is_deterministic_symmetric_and_preserves_diagonal():
    _, gram, _ = _problem()
    first = QuantizedSymmetricGram.from_dense(
        gram, block_size=5, ridge_lambda=100.0, error_fraction=0.1
    )
    second = QuantizedSymmetricGram.from_dense(
        gram, block_size=5, ridge_lambda=100.0, error_fraction=0.1
    )
    reconstructed = first.reconstruct()
    torch.testing.assert_close(reconstructed, reconstructed.T, rtol=0, atol=0)
    torch.testing.assert_close(reconstructed.diagonal(), gram.diagonal(), rtol=0, atol=0)
    assert first.bit_histogram() == second.bit_histogram()
    for left, right in zip(first.blocks, second.blocks):
        assert left.bits == right.bits
        assert torch.equal(left.values, right.values)
        assert torch.equal(left.scale, right.scale)


def test_streaming_requantization_error_is_bounded_against_exact_batch_gram():
    codes, _, _ = _problem(samples=60)
    first_codes, second_codes = codes[:23], codes[23:]
    exact_first = first_codes.T @ first_codes
    exact_final = exact_first + second_codes.T @ second_codes
    state = QuantizedSymmetricGram.from_dense(
        exact_first, block_size=6, ridge_lambda=100.0, error_fraction=0.1
    )
    state = state.merge(
        second_codes.T @ second_codes,
        ridge_lambda=100.0,
        error_fraction=0.1,
    )
    measured = float(torch.linalg.matrix_norm(state.reconstruct() - exact_final, ord=2))
    assert measured <= state.error_bound * (1 + 1e-10)
    assert state.merge_count == 2


def test_adaptive_precision_promotes_blocks_to_meet_budget():
    _, gram, _ = _problem(seed=307, dimension=24)
    int8 = QuantizedSymmetricGram.from_dense(
        gram, block_size=6, ridge_lambda=1e6, error_fraction=0.1
    )
    target = int8.error_bound * 0.60
    adaptive = QuantizedSymmetricGram.from_dense(
        gram,
        block_size=6,
        ridge_lambda=target / 0.1,
        error_fraction=0.1,
    )
    assert adaptive.bit_histogram()[16] > 0
    assert adaptive.error_bound <= target * (1 + 1e-6)


def test_fixed_int8_never_silently_promotes_or_relaxes_the_budget():
    _, gram, _ = _problem(seed=309, dimension=24)
    generous = QuantizedSymmetricGram.from_dense(
        gram,
        block_size=6,
        ridge_lambda=1e6,
        error_fraction=0.1,
        max_bits=8,
    )
    assert generous.bit_histogram()[16] == 0
    with torch.no_grad():
        try:
            QuantizedSymmetricGram.from_dense(
                gram,
                block_size=6,
                ridge_lambda=generous.error_bound / 0.2,
                error_fraction=0.1,
                max_bits=8,
            )
        except RuntimeError as error:
            assert "int8" in str(error) and "certificate" in str(error)
        else:
            raise AssertionError("fixed int8 must fail an impossible certificate")


def test_certified_solver_bounds_classifier_logits_and_argmax():
    codes, gram, cross = _problem(seed=311, dimension=20)
    ridge = 80.0
    state = QuantizedSymmetricGram.from_dense(
        gram, block_size=5, ridge_lambda=ridge, error_fraction=0.1
    )
    solution = solve_certified_ridge(state, cross, ridge, solve_dtype=DTYPE)
    exact = torch.linalg.solve(
        gram + ridge * torch.eye(len(gram), dtype=DTYPE), cross
    )
    classifier_error = float(torch.linalg.matrix_norm(solution.weights - exact, ord=2))
    assert classifier_error <= solution.absolute_classifier_error_bound * (1 + 1e-10)
    assert solution.relative_residual < 1e-12
    assert float(torch.linalg.eigvalsh(state.reconstruct() + ridge * torch.eye(len(gram), dtype=DTYPE)).min()) > 0

    query = codes[:16]
    exact_logits = query @ exact
    quantized_logits = query @ solution.weights
    bounds = logit_error_bound(query, solution)
    assert bool(((quantized_logits - exact_logits).abs().amax(dim=1) <= bounds * (1 + 1e-10)).all())
    certified = certified_argmax_mask(exact_logits, bounds)
    assert bool((exact_logits[certified].argmax(1) == quantized_logits[certified].argmax(1)).all())


def test_all_int8_projected_state_is_below_quarter_of_exact_fly():
    projection = projected_all_int8_state_bytes(
        feature_dim=768,
        expand_dim=10_000,
        synaptic_degree=300,
        num_classes=200,
        block_size=256,
    )
    assert projection["compressed_total_bytes"] < projection["exact_fly_total_bytes"]
    assert projection["state_fraction"] <= 0.25
    assert 100_000_000 < projection["compressed_total_bytes"] < 110_000_000
