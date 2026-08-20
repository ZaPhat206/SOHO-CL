"""Synthetic learner/state tests for CertiFLY."""

import inspect

import torch

from methods.certifly import CertiFLYLearner


def _learner(projection=None):
    return CertiFLYLearner(
        feature_dim=7,
        expand_dim=24,
        synaptic_degree=4,
        coding_level=0.25,
        block_size=6,
        error_fraction=0.1,
        max_bits=16,
        ridge_lower=2,
        ridge_upper=4,
        seed=2025,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
        projection=projection,
    )


def _stream():
    generator = torch.Generator().manual_seed(401)
    first = torch.randn(17, 24, generator=generator, dtype=torch.float64)
    second = torch.randn(13, 24, generator=generator, dtype=torch.float64)
    first_labels = torch.tensor([9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2])
    second_labels = torch.tensor([11, 5, 2, 11, 5, 2, 11, 5, 2, 11, 5, 2, 11])
    return first, first_labels, second, second_labels


def test_learner_is_exemplar_free_global_and_does_not_require_task_id():
    first, labels, _, _ = _stream()
    learner = _learner()
    learner.update_codes(first, labels, selected_ridge=100.0)
    assert learner.is_exemplar_free is True
    assert learner.class_ids == [2, 5, 9]
    assert learner.predict_logits_from_codes(first[:4]).shape == (4, 3)
    assert "task_id" not in inspect.signature(learner.predict_logits).parameters
    assert learner.diagnostics["gram_error_bound"] < learner.last_ridge
    learner.assert_exemplar_free_state()
    for name, tensor in learner.persistent_tensors().items():
        assert "history" not in name and "codes" not in name and "labels" not in name
        assert 17 not in tensor.shape


def test_checkpoint_resume_reproduces_logits_and_future_update():
    first, first_labels, second, second_labels = _stream()
    learner = _learner()
    learner.update_codes(first, first_labels, selected_ridge=100.0)
    logits = learner.predict_logits_from_codes(first[:5])

    clone = _learner()
    clone.load_state_dict(learner.state_dict())
    torch.testing.assert_close(clone.predict_logits_from_codes(first[:5]), logits, rtol=0, atol=0)
    assert clone.persistent_state_bytes() == learner.persistent_state_bytes()

    learner.update_codes(second, second_labels, selected_ridge=100.0)
    clone.update_codes(second, second_labels, selected_ridge=100.0)
    assert clone.class_ids == learner.class_ids == [2, 5, 9, 11]
    torch.testing.assert_close(
        clone.predict_logits_from_codes(second[:4]),
        learner.predict_logits_from_codes(second[:4]),
        rtol=0,
        atol=0,
    )
    assert clone.gram.error_bound == learner.gram.error_bound


def test_feature_path_returns_finite_logits_and_preserves_wta_sparsity():
    generator = torch.Generator().manual_seed(409)
    features = torch.randn(12, 7, generator=generator)
    labels = torch.tensor([0, 1, 2] * 4)
    learner = _learner()
    codes = learner.encode(features)
    assert codes.shape == (12, 24)
    assert bool(torch.isfinite(codes).all())
    assert torch.equal((codes != 0).sum(1), torch.full((12,), 6))
    learner.update(features, labels)
    logits = learner.predict_logits(features[:3])
    assert logits.shape == (3, 3) and bool(torch.isfinite(logits).all())


def test_checkpoint_configuration_mismatch_fails():
    first, labels, _, _ = _stream()
    learner = _learner()
    learner.update_codes(first, labels, selected_ridge=100.0)
    state = learner.state_dict()
    state["block_size"] = 12
    clone = _learner()
    try:
        clone.load_state_dict(state)
    except ValueError as error:
        assert "block_size" in str(error)
    else:
        raise AssertionError("configuration mismatch must fail")


def test_failed_certificate_update_is_transactional():
    first, first_labels, second, second_labels = _stream()
    learner = _learner()
    learner.update_codes(first, first_labels, selected_ridge=100.0)
    before = learner.state_dict()
    try:
        learner.update_codes(second * 1e6, second_labels, selected_ridge=1e-6)
    except RuntimeError as error:
        assert "certificate" in str(error)
    else:
        raise AssertionError("an impossible certificate budget must fail")
    after = learner.state_dict()
    assert after["class_ids"] == before["class_ids"]
    assert after["total_rows"] == before["total_rows"]
    torch.testing.assert_close(after["Q"], before["Q"], rtol=0, atol=0)
    torch.testing.assert_close(after["counts"], before["counts"], rtol=0, atol=0)
