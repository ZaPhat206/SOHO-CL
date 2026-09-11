"""Oracle and state-contract tests for the LoRanPAC continual-TSVD control."""

from __future__ import annotations

import math

import torch

from methods.frontends import RanPACAnalyticLearner
from methods.frontends.loranpac import (
    LoRanPACTSVBackend,
    maximum_loranpac_rank_for_backend_budget,
    official_loranpac_rank,
)


DTYPE = torch.float64


def _backend(*, dimension=11, truncate_percent=0.0, max_rank=11, ridge=2.0):
    return LoRanPACTSVBackend(
        dimension=dimension,
        truncate_percent=truncate_percent,
        max_rank=max_rank,
        ridge_lambda=ridge,
        statistics_dtype=DTYPE,
        solver_dtype=DTYPE,
    )


def test_official_rank_rule_reproduces_released_code_not_paper_ceil():
    # Released code: round(6 * .75) == 4; paper pseudocode: ceil(4.5) == 5.
    assert official_loranpac_rank(
        6, dimension=20, truncate_percent=25.0, max_rank=20
    ) == 4
    assert math.ceil(6 * 0.75) == 5
    assert official_loranpac_rank(
        100, dimension=17, truncate_percent=25.0, max_rank=9
    ) == 9


def test_full_rank_continual_update_matches_direct_ridge_projector():
    generator = torch.Generator().manual_seed(1301)
    first = torch.randn(8, 11, generator=generator, dtype=DTYPE)
    second = torch.randn(7, 11, generator=generator, dtype=DTYPE)
    labels_first = torch.tensor([2, 7, 2, 7, 2, 7, 2, 7])
    labels_second = torch.tensor([3, 7, 3, 7, 3, 7, 3])
    backend = _backend()
    backend.update(first, labels_first)
    backend.update(second, labels_second)

    features = torch.cat((first, second))
    labels = torch.cat((labels_first, labels_second))
    class_ids = sorted(set(labels.tolist()))
    columns = torch.tensor([class_ids.index(int(value)) for value in labels])
    targets = torch.nn.functional.one_hot(columns, num_classes=len(class_ids)).to(
        DTYPE
    )
    gram = features.T @ features
    cross = features.T @ targets
    expected = torch.linalg.solve(
        gram + 2.0 * torch.eye(11, dtype=DTYPE), cross
    )
    reconstructed = (backend.U * backend.s.unsqueeze(0)) @ (
        backend.U * backend.s.unsqueeze(0)
    ).T
    torch.testing.assert_close(reconstructed, gram, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(backend.Q, cross, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(backend.weights, expected, rtol=3e-12, atol=3e-12)
    assert backend.class_ids == class_ids
    assert backend.diagnostics["solver_relative_residual"] < 1e-12


def test_truncated_classifier_matches_loranpac_projected_equation():
    generator = torch.Generator().manual_seed(1302)
    features = torch.randn(12, 9, generator=generator, dtype=DTYPE)
    labels = torch.tensor([index % 3 for index in range(12)])
    backend = _backend(dimension=9, truncate_percent=25.0, max_rank=5, ridge=0.0)
    backend.update(features, labels)
    targets = torch.nn.functional.one_hot(labels, num_classes=3).to(DTYPE)
    cross = features.T @ targets
    expected = backend.U @ (
        (backend.U.T @ cross) / backend.s.square().unsqueeze(1)
    )
    torch.testing.assert_close(backend.weights, expected, rtol=2e-13, atol=2e-13)
    assert backend.effective_rank == 5
    matched, residual = backend.solve_weights(10.0)
    matched_expected = backend.U @ (
        (backend.U.T @ cross) / (backend.s.square() + 10.0).unsqueeze(1)
    )
    torch.testing.assert_close(matched, matched_expected, rtol=2e-13, atol=2e-13)
    assert residual < 1e-12


def test_state_budget_rank_is_symbolically_exact_and_never_exceeds_budget():
    dimension, classes, rank = 20, 4, 7
    base = 2 * dimension * classes * 4 + classes * 4
    target = base + rank * (dimension + 1) * 4
    assert maximum_loranpac_rank_for_backend_budget(
        dimension=dimension, num_classes=classes, target_bytes=target
    ) == rank
    assert maximum_loranpac_rank_for_backend_budget(
        dimension=dimension, num_classes=classes, target_bytes=target - 1
    ) == rank - 1


def test_ranpac_composition_checkpoint_and_exemplar_free_contract():
    generator = torch.Generator().manual_seed(1303)
    projection = torch.randn(6, 13, generator=generator, dtype=DTYPE)
    learner = RanPACAnalyticLearner(
        feature_dim=6,
        backend=_backend(dimension=13, truncate_percent=25.0, max_rank=6),
        projection=projection,
    )
    features = torch.randn(15, 6, generator=generator, dtype=DTYPE)
    labels = torch.tensor([index % 5 for index in range(15)])
    learner.update(features, labels)
    probe = torch.randn(7, 6, generator=generator, dtype=DTYPE)

    restored = RanPACAnalyticLearner(
        feature_dim=6,
        backend=_backend(dimension=13, truncate_percent=25.0, max_rank=6),
        projection=torch.zeros_like(projection),
    )
    restored.load_state_dict(learner.state_dict())
    torch.testing.assert_close(
        restored.predict_logits(probe), learner.predict_logits(probe), rtol=0, atol=0
    )
    assert restored.persistent_state_bytes() == learner.persistent_state_bytes()
    restored.assert_exemplar_free_state()
