import inspect

import pytest
import torch

from methods.zi_soho.learner import ZISOHOLearner


def learner(method="hurdle", seed=31):
    return ZISOHOLearner(
        raw_dim=6,
        expand_dim=12,
        synaptic_degree=3,
        coding_level=0.25,
        method=method,
        variance_kappa=3.0,
        score_chunk_size=2,
        seed=seed,
        dtype=torch.float64,
    )


@pytest.mark.parametrize(
    "method", ["wta_ncm", "support_only", "active_gaussian", "hurdle"]
)
def test_learner_is_global_finite_and_task_free(method):
    model = learner(method)
    features = torch.randn(45, 6, generator=torch.Generator().manual_seed(9))
    labels = torch.tensor([20, 4, 12] * 15)
    model.update(features[:17], labels[:17])
    model.update(features[17:], labels[17:])
    logits = model.predict_logits(features[:8])

    assert model.class_ids == [4, 12, 20]
    assert logits.shape == (8, 3)
    assert bool(torch.isfinite(logits).all())
    assert "task_id" not in inspect.signature(model.predict_logits).parameters
    model.assert_exemplar_free_state()


def test_checkpoint_contains_only_aggregate_state_and_roundtrips_logits():
    model = learner()
    features = torch.randn(60, 6, generator=torch.Generator().manual_seed(17))
    labels = torch.tensor([8, 1, 4] * 20)
    model.update(features, labels)
    expected = model.predict_logits(features[:9])
    state = model.state_dict()

    assert set(state["statistics"]) == {
        "feature_dim", "class_ids", "counts", "active_counts",
        "active_sums", "active_sq_sums",
    }
    assert not any(
        token in str(state).lower()
        for token in ("historical", "replay", "sample", "image")
    )
    restored = learner()
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.predict_logits(features[:9]), expected)


def test_persistent_shapes_scale_with_classes_not_observations():
    model = learner()
    labels = torch.tensor([0, 1, 2] * 20)
    model.update(torch.randn(60, 6, generator=torch.Generator().manual_seed(2)), labels)
    before = {name: tuple(value.shape) for name, value in model.persistent_tensors().items()}
    bytes_before = model.persistent_state_bytes()
    model.update(torch.randn(60, 6, generator=torch.Generator().manual_seed(3)), labels)
    after = {name: tuple(value.shape) for name, value in model.persistent_tensors().items()}

    assert before == after
    assert model.persistent_state_bytes() == bytes_before
    assert all(120 not in shape for shape in after.values())
    assert model.statistics.total_count == 120


def test_fixed_projection_and_sparse_codes_are_seed_deterministic():
    first, second = learner(seed=7), learner(seed=7)
    features = torch.randn(7, 6, generator=torch.Generator().manual_seed(5))
    first_indices, first_values = first.encode_sparse(features)
    second_indices, second_values = second.encode_sparse(features)

    torch.testing.assert_close(first.projection.to_dense(), second.projection.to_dense())
    torch.testing.assert_close(first_indices, second_indices)
    torch.testing.assert_close(first_values, second_values)


def test_checkpoint_configuration_mismatch_fails_closed():
    model = learner(seed=7)
    features = torch.randn(12, 6)
    model.update(features, torch.tensor([0, 1, 2] * 4))
    with pytest.raises(ValueError, match="seed"):
        learner(seed=8).load_state_dict(model.state_dict())
