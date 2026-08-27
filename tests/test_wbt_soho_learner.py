import inspect

import pytest
import torch

from methods.wbt_soho.learner import WBTMODES, WBTSOHOLearner


def _task(class_ids, *, samples=20, dimension=6):
    labels = torch.tensor(class_ids).repeat_interleave(samples)
    centers = torch.eye(dimension)[labels]
    generator = torch.Generator().manual_seed(307 + sum(class_ids))
    features = centers + 0.15 * torch.randn(
        len(labels), dimension, generator=generator
    )
    return torch.nn.functional.normalize(features, p=2, dim=1), labels


def _learner(mode):
    return WBTSOHOLearner(
        feature_dim=6,
        expand_dim=24,
        density=0.5,
        olda_dim=6,
        use_etf=True,
        coding_level=0.25,
        ridge_lambda=1.0,
        tangent_rank=3,
        pseudo_per_class=12,
        mode=mode,
        boundary_fraction=0.5,
        boundary_strength=0.5,
        seed=2025,
        dtype=torch.float64,
    )


@pytest.mark.parametrize("mode", sorted(WBTMODES))
def test_wbt_modes_stream_without_sample_state_and_without_task_id(mode):
    learner = _learner(mode)
    first_x, first_y = _task([0, 1])
    second_x, second_y = _task([2, 3])
    learner.update(first_x, first_y)
    learner.update(second_x, second_y)
    logits = learner.predict_logits(torch.cat((first_x[:3], second_x[:3])))
    assert logits.shape == (6, 4)
    assert torch.isfinite(logits).all()
    assert "task_id" not in inspect.signature(learner.predict_logits).parameters
    learner.assert_exemplar_free_state()
    assert learner.transport_memory.class_ids == [0, 1, 2, 3]
    assert not any(
        80 in tensor.shape for tensor in learner.persistent_tensors().values()
    )


def test_wbt_is_deterministic_and_rejects_repeated_classes():
    first, second = _learner("wta_boundary_transport"), _learner("wta_boundary_transport")
    for class_ids in ([0, 1], [2, 3]):
        features, labels = _task(class_ids)
        first.update(features, labels)
        second.update(features, labels)
    for name, value in first.persistent_tensors().items():
        assert torch.equal(value, second.persistent_tensors()[name])
    features, labels = _task([2, 3])
    with pytest.raises(ValueError, match="class-disjoint"):
        first.update(features, labels)
