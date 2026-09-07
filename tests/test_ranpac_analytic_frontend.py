"""Mathematical and state-contract tests for the RanPAC Phase-2 frontend."""

from __future__ import annotations

import inspect

import torch

from methods.analytic_ridge import ExactGramBackend
from methods.frontends import RanPACAnalyticLearner


def _new(projection=None):
    return RanPACAnalyticLearner(
        feature_dim=7,
        backend=ExactGramBackend(
            dimension=31,
            ridge_lambda=10.0,
            statistics_dtype=torch.float64,
            solver_dtype=torch.float64,
        ),
        seed=2025,
        projection=projection,
    )


def test_ranpac_frontend_matches_official_phase2_equation():
    learner = _new()
    generator = torch.Generator().manual_seed(51)
    first = torch.randn(17, 7, generator=generator, dtype=torch.float64)
    second = torch.randn(13, 7, generator=generator, dtype=torch.float64)
    labels_first = torch.tensor([2, 7, 11, 2, 7, 11, 2, 7, 11, 2, 7, 11, 2, 7, 11, 2, 7])
    labels_second = torch.tensor([3, 11, 7, 3, 11, 7, 3, 11, 7, 3, 11, 7, 3])
    learner.update(first, labels_first)
    learner.update(second, labels_second)

    features = torch.cat((first, second))
    labels = torch.cat((labels_first, labels_second))
    codes = torch.relu(features @ learner.projection)
    class_ids = sorted(set(labels.tolist()))
    columns = torch.tensor([class_ids.index(int(value)) for value in labels])
    targets = torch.nn.functional.one_hot(columns, num_classes=len(class_ids)).to(
        torch.float64
    )
    gram = codes.T @ codes
    cross = codes.T @ targets
    expected = torch.linalg.solve(
        gram + 10.0 * torch.eye(31, dtype=torch.float64), cross
    )
    torch.testing.assert_close(learner.backend.gram, gram, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(learner.Q, cross, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(learner.weights, expected, rtol=3e-13, atol=3e-13)
    assert learner.class_ids == class_ids


def test_ranpac_projection_is_seeded_dense_gaussian_and_fixed():
    first = _new()
    second = _new()
    assert first.projection.layout == torch.strided
    assert first.projection.shape == (7, 31)
    assert torch.equal(first.projection, second.projection)
    before = first.projection.clone()
    first.update(torch.ones(5, 7, dtype=torch.float64), torch.arange(5))
    assert torch.equal(first.projection, before)


def test_ranpac_checkpoint_round_trip_preserves_logits_and_bytes():
    first = _new()
    generator = torch.Generator().manual_seed(52)
    features = torch.randn(19, 7, generator=generator, dtype=torch.float64)
    labels = torch.tensor([index % 4 for index in range(19)])
    first.update(features, labels)
    restored = _new(projection=torch.zeros_like(first.projection))
    restored.load_state_dict(first.state_dict())
    probe = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(restored.predict_logits(probe), first.predict_logits(probe), rtol=0, atol=0)
    assert restored.persistent_state_bytes() == first.persistent_state_bytes()
    restored.assert_exemplar_free_state()


def test_ranpac_frontend_has_no_wta_or_flyhash_dependency():
    source = inspect.getsource(RanPACAnalyticLearner)
    assert "FlyHash" not in source
    assert "topk" not in source.lower()
    assert "coding_level" not in source
