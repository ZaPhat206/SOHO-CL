import inspect

import pytest
import torch

from methods.pps_soho.learner import PPSSOHOLearner


DTYPE = torch.float64


def learner(mode="class_protected", seed=23):
    return PPSSOHOLearner(
        raw_dim=6,
        anchor_dim=12,
        synaptic_degree=3,
        coding_level=0.25,
        sketch_size=4,
        ridge_lambda=0.3,
        gamma=0.7,
        mode=mode,
        seed=seed,
        dtype=DTYPE,
    )


@pytest.mark.parametrize("mode", ["class_protected", "global"])
def test_learner_is_global_finite_task_free_and_exemplar_free(mode):
    model = learner(mode)
    generator = torch.Generator().manual_seed(31)
    features = torch.randn(45, 6, generator=generator)
    labels = torch.tensor([20, 4, 12] * 15)
    model.update(features[:17], labels[:17])
    model.update(features[17:], labels[17:])
    logits = model.predict_logits(features[:9])

    assert logits.shape == (9, 3)
    assert bool(torch.isfinite(logits).all())
    assert model.class_ids == [4, 12, 20]
    assert "task_id" not in inspect.signature(model.predict_logits).parameters
    model.assert_exemplar_free_state()
    assert all(45 not in tensor.shape for tensor in model.persistent_tensors().values())


def test_checkpoint_has_only_aggregate_state_and_roundtrips_logits():
    model = learner()
    features = torch.randn(42, 6, generator=torch.Generator().manual_seed(5))
    labels = torch.tensor([8, 1, 4] * 14)
    model.update(features, labels)
    expected = model.predict_logits(features[:8])
    state = model.state_dict()

    assert set(state) == {
        "version", "raw_dim", "anchor_dim", "synaptic_degree", "coding_level",
        "sketch_size", "ridge_lambda", "gamma", "mode", "seed",
        "anchor_projection", "statistics",
    }
    assert not any(token in str(state).lower() for token in ("historical", "replay", "sample"))
    restored = learner()
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.predict_logits(features[:8]), expected, atol=1e-10, rtol=1e-10)


def test_checkpoint_configuration_mismatch_fails_closed():
    model = learner()
    model.update(torch.randn(12, 6), torch.tensor([0, 1, 2] * 4))
    with pytest.raises(ValueError, match="seed"):
        learner(seed=24).load_state_dict(model.state_dict())


def test_persistent_state_scales_with_dimensions_not_observation_count():
    model = learner()
    first = torch.randn(24, 6, generator=torch.Generator().manual_seed(9))
    labels = torch.tensor([0, 1, 2] * 8)
    model.update(first, labels)
    shapes_before = {
        name: tuple(value.shape) for name, value in model.persistent_tensors().items()
    }
    model.update(torch.randn(60, 6, generator=torch.Generator().manual_seed(10)), labels.repeat(3)[:60])
    shapes_after = {
        name: tuple(value.shape) for name, value in model.persistent_tensors().items()
    }

    assert shapes_after == shapes_before
    assert model.statistics.total_count == 84
