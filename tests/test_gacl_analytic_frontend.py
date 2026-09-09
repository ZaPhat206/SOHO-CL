"""Equation and frontend tests for the controlled GACL adapter."""

from __future__ import annotations

import math

import torch

from methods.analytic_ridge import ExactGramBackend
from methods.frontends import (
    GACLAnalyticLearner,
    GACLInverseRLSReference,
    gacl_linear_projection,
)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(expected).item()), 1.0)
    return float(torch.linalg.vector_norm(actual - expected).item()) / denominator


def test_projection_matches_seeded_bias_free_linear_initializer():
    seed = 812
    torch.manual_seed(seed)
    layer = torch.nn.Linear(7, 13, bias=False, dtype=torch.float64)
    projection = gacl_linear_projection(
        7, 13, seed=seed, dtype=torch.float64, device="cpu"
    )
    assert torch.equal(projection, layer.weight.T.contiguous())
    assert float(projection.abs().max()) <= 1.0 / math.sqrt(7)


def test_inverse_rls_primal_and_joint_match_with_reappearing_classes():
    generator = torch.Generator().manual_seed(913)
    dimension = 11
    gamma = 3.0
    batches = [
        (torch.randn(9, dimension, generator=generator, dtype=torch.float64), torch.tensor([4, 1, 4, 1, 4, 1, 4, 1, 4])),
        (torch.randn(8, dimension, generator=generator, dtype=torch.float64), torch.tensor([1, 7, 1, 7, 1, 7, 1, 7])),
        (torch.randn(7, dimension, generator=generator, dtype=torch.float64), torch.tensor([4, 7, 2, 4, 7, 2, 2])),
        (torch.randn(6, dimension, generator=generator, dtype=torch.float64), torch.tensor([1, 2, 4, 7, 2, 1])),
    ]
    reference = GACLInverseRLSReference(
        dimension=dimension, gamma=gamma, dtype=torch.float64
    )
    exact = ExactGramBackend(
        dimension=dimension,
        ridge_lambda=gamma,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
    )
    all_codes, all_labels = [], []
    seen: set[int] = set()
    for codes, labels in batches:
        old = set(seen)
        seen.update(map(int, labels.tolist()))
        assert old & set(map(int, labels.tolist())) or not old
        reference.update_codes(codes, labels)
        exact.update(codes, labels)
        all_codes.append(codes)
        all_labels.append(labels)
        assert reference.class_ids == exact.class_ids
        assert _relative_error(reference.weights, exact.weights) < 2e-12

        stacked_codes = torch.cat(all_codes)
        stacked_labels = torch.cat(all_labels)
        columns = torch.tensor(
            [exact.class_ids.index(int(value)) for value in stacked_labels]
        )
        targets = torch.nn.functional.one_hot(
            columns, num_classes=len(exact.class_ids)
        ).to(torch.float64)
        system = stacked_codes.T @ stacked_codes
        system.diagonal().add_(gamma)
        joint = torch.linalg.solve(system, stacked_codes.T @ targets)
        assert _relative_error(reference.weights, joint) < 2e-12


def test_gacl_frontend_delegates_repeated_labels_and_is_exemplar_free():
    backend = ExactGramBackend(
        dimension=17,
        ridge_lambda=10.0,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
    )
    learner = GACLAnalyticLearner(feature_dim=5, backend=backend, seed=2025)
    generator = torch.Generator().manual_seed(19)
    learner.update(torch.randn(12, 5, generator=generator), torch.tensor([3, 8] * 6))
    learner.update(torch.randn(10, 5, generator=generator), torch.tensor([8, 1] * 5))
    assert learner.class_ids == [1, 3, 8]
    assert learner.weights is not None
    assert learner.weights.shape == (17, 3)
    learner.assert_exemplar_free_state()
    assert learner.persistent_state_bytes() > backend.persistent_state_bytes()
