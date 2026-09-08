"""Tests for the fixed signed-hash analytic frontend."""

import torch

from methods.analytic_ridge import ExactGramBackend, persistent_tensor_bytes
from methods.frontends import CountSketchAnalyticLearner


def test_countsketch_matches_manual_signed_bucket_sum():
    backend = ExactGramBackend(dimension=3, ridge_lambda=2.0)
    learner = CountSketchAnalyticLearner(
        input_dimension=5,
        backend=backend,
        bucket_indices=torch.tensor([0, 1, 0, 2, 1], dtype=torch.int32),
        signs=torch.tensor([1, -1, -1, 1, 1], dtype=torch.int8),
    )
    codes = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    expected = torch.tensor([[-2.0, 3.0, 4.0]])
    assert torch.equal(learner.encode_codes(codes), expected)


def test_countsketch_is_deterministic_and_counts_all_persistent_tensors():
    first = CountSketchAnalyticLearner(
        input_dimension=17,
        backend=ExactGramBackend(dimension=7, ridge_lambda=1.0),
        seed=2025,
    )
    second = CountSketchAnalyticLearner(
        input_dimension=17,
        backend=ExactGramBackend(dimension=7, ridge_lambda=1.0),
        seed=2025,
    )
    assert torch.equal(first.bucket_indices, second.bucket_indices)
    assert torch.equal(first.signs, second.signs)
    features = torch.randn(12, 17, generator=torch.Generator().manual_seed(9))
    labels = torch.tensor([index % 3 for index in range(12)])
    first.update_codes(features, labels)
    first.assert_exemplar_free_state()
    assert first.persistent_state_bytes() == persistent_tensor_bytes(
        first.persistent_tensors()
    )
    assert first.bucket_indices.dtype == torch.int32
    assert first.signs.dtype == torch.int8


def test_countsketch_rejects_invalid_mapping():
    backend = ExactGramBackend(dimension=2, ridge_lambda=1.0)
    try:
        CountSketchAnalyticLearner(
            input_dimension=3,
            backend=backend,
            bucket_indices=torch.tensor([0, 1, 2]),
            signs=torch.tensor([1, 1, 1]),
        )
    except ValueError as error:
        assert "out of range" in str(error)
    else:
        raise AssertionError("out-of-range CountSketch bucket was accepted")
