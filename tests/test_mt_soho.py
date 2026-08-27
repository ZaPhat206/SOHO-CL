import inspect

import pytest
import torch

from methods.mt_soho import MTSOHOLearner
from methods.mt_soho.geometry import class_geometry, transport_moments


def _stream(dtype=torch.float64):
    generator = torch.Generator().manual_seed(17)
    x0 = torch.randn((24, 6), generator=generator, dtype=dtype)
    y0 = torch.arange(2).repeat_interleave(12)
    x0 += torch.nn.functional.one_hot(y0, 6).to(dtype) * 1.5
    x1 = torch.randn((24, 6), generator=generator, dtype=dtype)
    y1 = torch.arange(2, 4).repeat_interleave(12)
    x1 += torch.nn.functional.one_hot(y1, 6).to(dtype) * 1.5
    return (x0, y0), (x1, y1)


def _learner(method="mt_whitened", seed=2025):
    return MTSOHOLearner(
        method=method,
        feature_dim=6,
        expand_dim=18,
        synaptic_degree=3,
        coding_level=1 / 3,
        anchor_ridge=0.7,
        projection_ridge=0.5,
        adapted_ridge=0.3,
        target_rank=3,
        shrinkage=0.2,
        adaptation_weight=0.5,
        geometry_epsilon=1e-8,
        seed=seed,
        dtype=torch.float64,
    )


def test_fixed_anchor_streaming_statistics_equal_batch_statistics():
    learner = _learner("fixed_wta_ridge")
    stream = _stream()
    encoded, raw, labels = [], [], []
    for x, y in stream:
        encoded.append(learner._encode(x))
        raw.append(x)
        labels.append(y)
        learner.update(x, y)
    u, x, y = torch.cat(encoded), torch.cat(raw), torch.cat(labels)
    targets = torch.nn.functional.one_hot(y, num_classes=4).to(torch.float64)
    stats = learner.statistics
    assert torch.allclose(stats.G_u, u.T @ u, atol=1e-10, rtol=1e-10)
    assert torch.allclose(stats.Q_u, u.T @ targets, atol=1e-10, rtol=1e-10)
    assert torch.allclose(stats.G_x, x.T @ x, atol=1e-10, rtol=1e-10)
    assert torch.allclose(stats.Q_x, x.T @ targets, atol=1e-10, rtol=1e-10)


def test_transport_moments_equal_explicit_batch_reprojection():
    learner = _learner()
    stream = _stream()
    all_x, all_y = [], []
    for x, y in stream:
        all_x.append(x)
        all_y.append(y)
        learner.update(x, y)
    x, y = torch.cat(all_x), torch.cat(all_y)
    u = learner._encode(x)
    v = u @ learner.transport
    targets = torch.nn.functional.one_hot(y, num_classes=4).to(torch.float64)
    gram_v, cross_v = transport_moments(
        learner.statistics.G_u, learner.statistics.Q_u, learner.transport
    )
    assert torch.allclose(gram_v, v.T @ v, atol=1e-9, rtol=1e-9)
    assert torch.allclose(cross_v, v.T @ targets, atol=1e-9, rtol=1e-9)


def test_streaming_logits_equal_explicit_batch_oracle_after_transport_changes():
    learner = _learner()
    all_x, all_y = [], []
    for x, y in _stream():
        all_x.append(x)
        all_y.append(y)
        learner.update(x, y)
    x, y = torch.cat(all_x), torch.cat(all_y)
    u = learner._encode(x)
    targets = torch.nn.functional.one_hot(y, num_classes=4).to(torch.float64)
    anchor = torch.linalg.solve(
        u.T @ u + learner.anchor_ridge * torch.eye(18, dtype=torch.float64),
        u.T @ targets,
    )
    v = u @ learner.transport
    adapted = torch.linalg.solve(
        v.T @ v + learner.adapted_ridge * torch.eye(v.shape[1], dtype=torch.float64),
        v.T @ targets,
    )
    expected = u @ anchor + learner.adaptation_weight * (v @ adapted)
    assert torch.allclose(learner.predict_logits(x), expected, atol=1e-8, rtol=1e-8)


def test_raw_moments_reconstruct_exact_pooled_within_scatter():
    learner = _learner()
    all_x, all_y = [], []
    for x, y in _stream():
        all_x.append(x)
        all_y.append(y)
        learner.update(x, y)
    x, y = torch.cat(all_x), torch.cat(all_y)
    expected = torch.zeros((6, 6), dtype=torch.float64)
    for class_id in range(4):
        values = x[y == class_id]
        centered = values - values.mean(0)
        expected += centered.T @ centered
    stats = learner.statistics
    observed = stats.G_x - (stats.Q_x / stats.counts.unsqueeze(0)) @ stats.Q_x.T
    assert torch.allclose(observed, expected, atol=1e-9, rtol=1e-9)


def test_geometry_is_deterministic_finite_and_strictly_class_centered_rank():
    learner = _learner()
    for x, y in _stream():
        learner.update(x, y)
    stats = learner.statistics
    first, diagnostics = class_geometry(
        raw_gram=stats.G_x,
        raw_cross=stats.Q_x,
        counts=stats.counts,
        requested_rank=99,
        shrinkage=0.2,
        epsilon=1e-8,
        whiten=True,
    )
    second, _ = class_geometry(
        raw_gram=stats.G_x,
        raw_cross=stats.Q_x,
        counts=stats.counts,
        requested_rank=99,
        shrinkage=0.2,
        epsilon=1e-8,
        whiten=True,
    )
    assert first.shape == (4, 3)
    assert diagnostics["effective_rank"] == 3
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)
    assert torch.allclose(first.norm(dim=1), torch.ones(4, dtype=torch.float64))


def test_shuffled_control_changes_targets_but_not_anchor_statistics():
    proposed, shuffled = _learner("mt_whitened"), _learner("mt_shuffled")
    for x, y in _stream():
        proposed.update(x, y)
        shuffled.update(x, y)
    assert torch.equal(proposed.statistics.G_u, shuffled.statistics.G_u)
    assert torch.equal(proposed.statistics.Q_u, shuffled.statistics.Q_u)
    assert not torch.equal(proposed.targets, shuffled.targets)


def test_checkpoint_roundtrip_is_sample_free_and_preserves_logits():
    learner = _learner()
    for x, y in _stream():
        learner.update(x, y)
    state = learner.state_dict()
    serialized_names = repr(state).lower()
    assert "history" not in serialized_names and "replay" not in serialized_names
    restored = _learner()
    restored.load_state_dict(state)
    probe = torch.randn((7, 6), generator=torch.Generator().manual_seed(71), dtype=torch.float64)
    assert torch.allclose(restored.predict_logits(probe), learner.predict_logits(probe))
    assert set(restored.persistent_tensors()) == set(learner.persistent_tensors())


def test_state_has_no_historical_sample_dimension_and_inference_has_no_task_id():
    learner = _learner()
    for x, y in _stream():
        learner.update(x, y)
    learner.assert_exemplar_free_state()
    assert "task_id" not in inspect.signature(learner.predict_logits).parameters
    for name, tensor in learner.persistent_tensors().items():
        assert 48 not in tensor.shape, name


def test_invalid_configuration_fails_closed():
    with pytest.raises(ValueError):
        MTSOHOLearner(
            method="unknown", feature_dim=6, expand_dim=18,
            synaptic_degree=3, coding_level=0.2, anchor_ridge=1.0,
        )
