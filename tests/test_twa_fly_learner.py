import inspect

import pytest
import torch

from methods.cached_replay_baselines import CachedFlyCL
from methods.twa_fly import TWAFLYLearner


DTYPE = torch.float64


def learner(method="twa_symmetric", rho=0.2, seed=13):
    return TWAFLYLearner(
        method=method,
        raw_dim=6,
        fly_dim=12,
        num_classes=4,
        synaptic_degree=3,
        coding_level=0.25,
        rho=rho,
        raw_ridge=0.3,
        fly_ridge=0.7,
        solver_tolerance=1e-10,
        solver_max_iterations=300,
        seed=seed,
        dtype=DTYPE,
    )


def stream(seed=31):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(40, 6, generator=generator)
    labels = torch.tensor([0, 1, 2, 3] * 10)
    return x, labels


@pytest.mark.parametrize("method", ["twa_one_way", "twa_symmetric", "twa_shuffled_cross"])
def test_methods_are_global_finite_task_id_free_and_exemplar_free(method):
    model = learner(method)
    x, labels = stream()
    model.update(x[:17], labels[:17])
    model.update(x[17:], labels[17:])
    logits = model.predict_logits(x[:9])
    assert logits.shape == (9, 4)
    assert bool(torch.isfinite(logits).all())
    assert "task_id" not in inspect.signature(model.predict_logits).parameters
    model.assert_exemplar_free_state()
    assert model.statistics.total_count == 40
    assert all(40 not in tensor.shape for tensor in model.persistent_tensors().values())


def test_rho_zero_learner_logits_equal_matched_cached_fly():
    x, labels = stream()
    twa = TWAFLYLearner(
        method="twa_symmetric", raw_dim=6, fly_dim=12, num_classes=4,
        synaptic_degree=3, coding_level=0.25, rho=0.0, raw_ridge=0.3,
        fly_ridge=0.7, seed=13, dtype=torch.float32,
    )
    fly = CachedFlyCL(
        feature_dim=6,
        expand_dim=12,
        synaptic_degree=3,
        coding_level=0.25,
        ridge_lambda=0.7,
        seed=13,
        dtype=torch.float32,
    )
    for start, stop in ((0, 17), (17, 40)):
        twa.update(x[start:stop], labels[start:stop])
        fly.update(x[start:stop], labels[start:stop])
    torch.testing.assert_close(
        twa.flyhash.projection_matrix.to_dense(), fly.flyhash.projection_matrix.to_dense(),
        atol=0, rtol=0,
    )
    torch.testing.assert_close(twa.statistics.G_zz, fly.statistics.G, atol=0, rtol=0)
    torch.testing.assert_close(twa.statistics.Q_z, fly.statistics.Q, atol=0, rtol=0)
    torch.testing.assert_close(twa.predict_logits(x), fly.predict_logits(x), atol=1e-5, rtol=1e-5)


def test_checkpoint_contains_aggregate_state_only_and_roundtrips_logits():
    original = learner()
    x, labels = stream()
    original.update(x, labels)
    expected = original.predict_logits(x[:7])
    state = original.state_dict()
    assert set(state["statistics"]) == {
        "raw_dim", "fly_dim", "num_classes", "G_xx", "G_zz", "R_xz", "Q_x", "Q_z", "counts"
    }
    assert not any(token in str(state).lower() for token in ("sample", "replay", "historical", "feature_history"))
    restored = learner()
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.predict_logits(x[:7]), expected, atol=1e-10, rtol=1e-10)


def test_checkpoint_configuration_mismatch_fails_closed():
    original = learner()
    x, labels = stream()
    original.update(x, labels)
    with pytest.raises(ValueError, match="seed"):
        learner(seed=14).load_state_dict(original.state_dict())
