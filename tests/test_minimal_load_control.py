"""Correctness gates for the uncertified minimal-load direct-Gram control."""

import pytest
import torch

from methods.srq_fly_optimized.minimal_load_control import (
    MinimalLoadDirectInt8GramLearner,
    _cholesky_succeeds,
    grid_load,
    minimal_cholesky_load,
)


def _indefinite(dimension=12, smallest=-5.0, dtype=torch.float64):
    generator = torch.Generator().manual_seed(4)
    basis, _ = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator, dtype=dtype))
    eigenvalues = torch.linspace(smallest, 50.0, dimension, dtype=dtype)
    matrix = basis @ torch.diag(eigenvalues) @ basis.T
    return (matrix + matrix.T) * 0.5


def test_search_returns_smallest_successful_grid_point_and_restores_matrix():
    matrix = _indefinite()
    before = matrix.clone()
    result = minimal_cholesky_load(
        matrix, ridge_lambda=1.0, base_load=0.1, steps_per_doubling=4, maximum_load=1e4
    )
    assert torch.equal(matrix, before)
    # Success needs 1 + load > 5: 0.1 * 2**(22/4) = 4.53 works, 0.1 * 2**(21/4) = 3.81 does not.
    assert result["grid_index"] == 22 and result["zero_load_succeeded"] is False
    assert result["load"] == pytest.approx(grid_load(0.1, 22, 4))
    loaded = matrix.clone()
    loaded.diagonal().add_(1.0 + result["load"])
    below = matrix.clone()
    below.diagonal().add_(1.0 + grid_load(0.1, 21, 4))
    assert _cholesky_succeeds(loaded) and not _cholesky_succeeds(below)
    assert result["attempts"] <= 1 + 7 + 4


def test_search_uses_zero_load_for_a_definite_matrix():
    matrix = _indefinite(smallest=2.0)
    result = minimal_cholesky_load(
        matrix, ridge_lambda=1.0, base_load=0.1, steps_per_doubling=4, maximum_load=1e4
    )
    assert result == {"load": 0.0, "grid_index": None, "attempts": 1, "zero_load_succeeded": True}


def test_search_refuses_to_exceed_its_maximum_load():
    matrix = _indefinite(smallest=-1e6)
    before = matrix.clone()
    with pytest.raises(RuntimeError):
        minimal_cholesky_load(
            matrix, ridge_lambda=1.0, base_load=0.1, steps_per_doubling=4, maximum_load=100.0
        )
    assert torch.equal(matrix, before)


def _kwargs(**overrides):
    kwargs = {
        "feature_dim": 7, "expand_dim": 18, "synaptic_degree": 3, "coding_level": 1 / 3,
        "ridge_lambda": 1e-3, "block_size": 7, "group_size": 5, "seed": 2025,
        "device": "cpu", "statistics_dtype": torch.float32, "solver_dtype": torch.float32,
    }
    kwargs.update(overrides)
    return kwargs


def _stream():
    generator = torch.Generator().manual_seed(8102)
    first = torch.randn(17, 7, generator=generator)
    second = 1.7 * torch.randn(13, 7, generator=generator)
    return first, torch.arange(17) % 3, second, 3 + torch.arange(13) % 3


def test_learner_solves_its_loaded_system_and_round_trips():
    learner = MinimalLoadDirectInt8GramLearner(**_kwargs())
    first, first_labels, second, second_labels = _stream()
    for features, labels in ((first, first_labels), (second, second_labels)):
        learner.update(features, labels)
        loading = float(learner.diagonal_loading)
        assert loading >= 0.0 and learner.diagnostics["cholesky_attempts"] >= 1
        system = learner.gram.reconstruct_symmetric(dtype=torch.float32)
        system.diagonal().add_(learner.ridge_lambda + loading)
        residual = torch.linalg.vector_norm(system @ learner.weights - learner.Q)
        assert float(residual) <= 1e-3 * max(float(torch.linalg.vector_norm(learner.Q)), 1.0)
        assert "diagonal_loading" in learner.persistent_tensors()
        learner.assert_exemplar_free_state()
    restored = MinimalLoadDirectInt8GramLearner(**_kwargs())
    restored.load_state_dict(learner.state_dict())
    assert torch.equal(restored.weights, learner.weights)
    assert float(restored.diagonal_loading) == float(learner.diagonal_loading)


def test_learner_rejects_mismatched_checkpoint_policy():
    learner = MinimalLoadDirectInt8GramLearner(**_kwargs())
    first, first_labels, _, _ = _stream()
    learner.update(first, first_labels)
    other = MinimalLoadDirectInt8GramLearner(load_grid_steps_per_doubling=2, **_kwargs())
    with pytest.raises(ValueError):
        other.load_state_dict(learner.state_dict())
