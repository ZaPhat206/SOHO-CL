"""Synthetic learner/state tests for the SRQ-FLY D0 diagnostic."""

import inspect

import pytest
import torch

from methods.srq_fly import DirectInt8GramLearner, SquareRootFLYLearner


def _kwargs(projection=None):
    return dict(
        feature_dim=7,
        expand_dim=24,
        synaptic_degree=4,
        coding_level=0.25,
        ridge_lambda=100.0,
        block_size=6,
        group_size=5,
        seed=2025,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
        projection=projection,
    )


def _learner(kind: str):
    if kind == "direct":
        return DirectInt8GramLearner(**_kwargs())
    return SquareRootFLYLearner(storage_mode=kind, **_kwargs())


def _stream():
    generator = torch.Generator().manual_seed(709)
    first = torch.randn(17, 24, generator=generator, dtype=torch.float64)
    second = torch.randn(13, 24, generator=generator, dtype=torch.float64)
    first_labels = torch.tensor([9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2])
    second_labels = torch.tensor([11, 5, 2, 11, 5, 2, 11, 5, 2, 11, 5, 2, 11])
    return first, first_labels, second, second_labels


@pytest.mark.parametrize("kind", ["direct", "float16", "int8"])
def test_learners_are_global_finite_and_exemplar_free(kind):
    first, labels, _, _ = _stream()
    learner = _learner(kind)
    learner.update_codes(first, labels)
    assert learner.is_exemplar_free is True
    assert learner.class_ids == [2, 5, 9]
    logits = learner.predict_logits_from_codes(first[:4])
    assert logits.shape == (4, 3) and bool(torch.isfinite(logits).all())
    assert "task_id" not in inspect.signature(learner.predict_logits).parameters
    assert learner.diagnostics["solver_relative_residual"] < 1e-10
    learner.assert_exemplar_free_state()
    for name, tensor in learner.persistent_tensors().items():
        assert "history" not in name and "codes" not in name and "labels" not in name
        assert 17 not in tensor.shape


@pytest.mark.parametrize("kind", ["direct", "float16", "int8"])
def test_checkpoint_resume_reproduces_logits_and_future_updates(kind):
    first, first_labels, second, second_labels = _stream()
    learner = _learner(kind)
    learner.update_codes(first, first_labels)
    expected = learner.predict_logits_from_codes(first[:5])
    clone = _learner(kind)
    clone.load_state_dict(learner.state_dict())
    torch.testing.assert_close(clone.predict_logits_from_codes(first[:5]), expected, rtol=0, atol=0)
    assert clone.persistent_state_bytes() == learner.persistent_state_bytes()
    learner.update_codes(second, second_labels)
    clone.update_codes(second, second_labels)
    assert clone.class_ids == learner.class_ids == [2, 5, 9, 11]
    torch.testing.assert_close(
        clone.predict_logits_from_codes(second[:4]),
        learner.predict_logits_from_codes(second[:4]),
        rtol=0,
        atol=0,
    )


def test_feature_path_preserves_fly_sparsity_and_is_finite():
    generator = torch.Generator().manual_seed(719)
    features = torch.randn(12, 7, generator=generator)
    labels = torch.tensor([0, 1, 2] * 4)
    learner = _learner("int8")
    codes = learner.encode(features)
    assert codes.shape == (12, 24)
    assert bool(torch.isfinite(codes).all())
    assert torch.equal((codes != 0).sum(1), torch.full((12,), 6))
    learner.update(features, labels)
    assert learner.predict_logits(features[:3]).shape == (3, 3)


def test_streaming_square_root_matches_its_decoded_batch_system():
    first, first_labels, second, second_labels = _stream()
    learner = _learner("int8")
    learner.update_codes(first, first_labels)
    learner.update_codes(second, second_labels)
    factor = learner.factor.reconstruct_upper(dtype=torch.float64)
    oracle = torch.linalg.solve(factor.T @ factor, learner.Q)
    torch.testing.assert_close(learner.weights, oracle, rtol=1e-10, atol=1e-10)


def test_invalid_update_is_transactional():
    first, labels, _, _ = _stream()
    learner = _learner("int8")
    learner.update_codes(first, labels)
    before = learner.state_dict()
    invalid = first[:2].clone()
    invalid[0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN"):
        learner.update_codes(invalid, labels[:2])
    after = learner.state_dict()
    assert after["total_rows"] == before["total_rows"]
    assert after["class_ids"] == before["class_ids"]
    torch.testing.assert_close(after["Q"], before["Q"], rtol=0, atol=0)


def test_checkpoint_configuration_mismatch_fails():
    first, labels, _, _ = _stream()
    learner = _learner("int8")
    learner.update_codes(first, labels)
    state = learner.state_dict()
    state["factor"]["group_size"] = 9
    with pytest.raises(ValueError, match="compressed checkpoint|invalid int8"):
        _learner("int8").load_state_dict(state)
