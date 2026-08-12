"""Synthetic mathematical and state-contract tests for SFT-CL."""

import inspect

import torch

from methods.sft_cl import create_learner
from methods.sft_cl.geometry import (
    analytic_confusion_affinity,
    confusion_between_scatter,
    fisher_transport,
    raw_ridge_weights,
    scatter_matrices,
    shuffled_affinity,
)
from methods.sft_cl.statistics import FixedFeatureStatistics


DTYPE = torch.float64
TOL = 1e-9


def _stream():
    torch.manual_seed(314)
    x = torch.randn(30, 7, dtype=DTYPE)
    y = torch.tensor([7, 2, 11, 7, 2, 11, 13, 7, 13, 2] * 3)
    return x, y


def _statistics():
    x, y = _stream()
    statistics = FixedFeatureStatistics(x.shape[1], dtype=DTYPE)
    statistics.update(x[:9], y[:9])
    statistics.update(x[9:21], y[9:21])
    statistics.update(x[21:], y[21:])
    return x, y, statistics


def _targets(labels, class_ids):
    columns = torch.tensor([class_ids.index(int(label)) for label in labels])
    return torch.nn.functional.one_hot(columns, num_classes=len(class_ids)).to(DTYPE)


def test_streaming_statistics_and_scatter_equal_batch_oracle():
    x, y, statistics = _statistics()
    targets = _targets(y, statistics.class_ids)
    torch.testing.assert_close(statistics.G, x.T @ x, rtol=0, atol=TOL)
    torch.testing.assert_close(statistics.Q, x.T @ targets, rtol=0, atol=TOL)
    torch.testing.assert_close(statistics.counts, targets.sum(0), rtol=0, atol=TOL)

    within, between, means = scatter_matrices(statistics)
    explicit_within = torch.zeros_like(within)
    global_mean = x.mean(0)
    explicit_between = torch.zeros_like(between)
    for column, class_id in enumerate(statistics.class_ids):
        members = x[y == class_id]
        centered = members - members.mean(0)
        explicit_within += centered.T @ centered
        offset = (members.mean(0) - global_mean).unsqueeze(1)
        explicit_between += len(members) * (offset @ offset.T)
        torch.testing.assert_close(means[:, column], members.mean(0), rtol=0, atol=TOL)
    torch.testing.assert_close(within, explicit_within, rtol=0, atol=TOL)
    torch.testing.assert_close(between, explicit_between, rtol=0, atol=TOL)


def test_late_lower_global_class_id_reorders_statistics_canonically():
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=DTYPE)
    statistics = FixedFeatureStatistics(2, dtype=DTYPE)
    statistics.update(x[:2], torch.tensor([9, 9]))
    statistics.update(x[2:], torch.tensor([3]))
    assert statistics.class_ids == [3, 9]
    expected = torch.tensor([[3.0, 1.0], [0.0, 2.0]], dtype=DTYPE)
    torch.testing.assert_close(statistics.Q, expected, rtol=0, atol=TOL)
    torch.testing.assert_close(statistics.counts, torch.tensor([1.0, 2.0], dtype=DTYPE), rtol=0, atol=TOL)


def test_transformed_sufficient_statistics_equal_explicit_reprojection():
    x, y, statistics = _statistics()
    within, between, _ = scatter_matrices(statistics)
    transport, _ = fisher_transport(within, between, statistics.total_count, 1e-4, mode="soft", kappa=.3, delta=.05)
    targets = _targets(y, statistics.class_ids)
    z = x @ transport
    torch.testing.assert_close(transport.T @ statistics.G @ transport, z.T @ z, rtol=0, atol=TOL)
    torch.testing.assert_close(transport.T @ statistics.Q, z.T @ targets, rtol=0, atol=TOL)


def test_nonorthogonal_transport_equals_anisotropic_ridge():
    x, _, statistics = _statistics()
    ridge = .7
    torch.manual_seed(9)
    transport = torch.randn(7, 7, dtype=DTYPE)
    transport += 2.0 * torch.eye(7, dtype=DTYPE)
    transformed = transport @ torch.linalg.solve(
        transport.T @ statistics.G @ transport + ridge * torch.eye(7, dtype=DTYPE),
        transport.T @ statistics.Q,
    )
    inverse_metric = torch.linalg.solve(transport @ transport.T, torch.eye(7, dtype=DTYPE))
    anisotropic = torch.linalg.solve(statistics.G + ridge * inverse_metric, statistics.Q)
    torch.testing.assert_close(transformed, anisotropic, rtol=0, atol=TOL)
    assert not torch.allclose(transport @ transport.T, torch.eye(7, dtype=DTYPE), rtol=0, atol=TOL)
    assert torch.isfinite(x @ transformed).all()


def test_orthogonal_transport_is_raw_isotropic_ridge_noop():
    _, _, statistics = _statistics()
    ridge = .25
    torch.manual_seed(10)
    transport, _ = torch.linalg.qr(torch.randn(7, 7, dtype=DTYPE))
    transported = transport @ torch.linalg.solve(
        transport.T @ statistics.G @ transport + ridge * torch.eye(7, dtype=DTYPE),
        transport.T @ statistics.Q,
    )
    raw = raw_ridge_weights(statistics, ridge)
    torch.testing.assert_close(transported, raw, rtol=0, atol=TOL)


def test_soft_transport_is_full_rank_and_finite():
    _, _, statistics = _statistics()
    within, between, _ = scatter_matrices(statistics)
    transport, geometry = fisher_transport(within, between, statistics.total_count, 1e-4, mode="soft", kappa=.01, delta=.02)
    assert transport.shape == (7, 7)
    assert geometry.effective_rank == 7
    assert bool(torch.isfinite(transport).all())
    assert float(geometry.gains.min()) >= 0.02**0.5 - 1e-12
    assert torch.linalg.matrix_rank(transport).item() == 7


def test_confusion_affinity_and_shuffled_control_are_symmetric_deterministic():
    _, _, statistics = _statistics()
    within, _, means = scatter_matrices(statistics)
    affinity = analytic_confusion_affinity(means, within, statistics.counts, raw_ridge_weights(statistics, .2), 1e-5)
    shuffled_a = shuffled_affinity(affinity, seed=88)
    shuffled_b = shuffled_affinity(affinity, seed=88)
    torch.testing.assert_close(affinity, affinity.T, rtol=0, atol=TOL)
    torch.testing.assert_close(torch.diag(affinity), torch.zeros(4, dtype=DTYPE), rtol=0, atol=TOL)
    torch.testing.assert_close(shuffled_a, shuffled_b, rtol=0, atol=0)
    torch.testing.assert_close(torch.sort(shuffled_a[torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)]).values,
                               torch.sort(affinity[torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)]).values,
                               rtol=0, atol=TOL)
    weighted = confusion_between_scatter(means, statistics.counts, affinity)
    torch.testing.assert_close(weighted, weighted.T, rtol=0, atol=TOL)


def test_all_sft_methods_are_global_task_free_and_exemplar_free():
    x, y, _ = _statistics()
    methods = ("raw_ridge", "fisher_hard", "confusion_fisher_hard", "fisher_soft", "confusion_fisher_soft", "shuffled_confusion_fisher_soft")
    for method in methods:
        learner = create_learner(method=method, feature_dim=7, ridge_lambda=.2, requested_rank=3, kappa=.5, delta=.1, seed=27)
        learner.update(x[:15], y[:15])
        learner.update(x[15:], y[15:])
        logits = learner.predict_logits(x[:4])
        assert logits.shape == (4, 4)
        assert bool(torch.isfinite(logits).all())
        assert "task_id" not in inspect.signature(learner.predict_logits).parameters
        learner.assert_exemplar_free_state()
        assert learner.persistent_state_bytes() > 0
        checkpoint = learner.state_dict()
        serialized = repr(checkpoint).lower()
        assert "memory_features" not in serialized and "replay" not in serialized
        clone = create_learner(method=method, feature_dim=7, ridge_lambda=.2, requested_rank=3, kappa=.5, delta=.1, seed=27)
        clone.load_state_dict(checkpoint)
        torch.testing.assert_close(clone.predict_logits(x[:4]), logits, rtol=0, atol=TOL)
