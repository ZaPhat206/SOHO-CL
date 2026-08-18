import inspect

import pytest
import torch

from methods.cars_fly import CARSFLYLearner
from methods.crt_soho import CRTSOHOLearner


DTYPE = torch.float64


def learner(**overrides):
    arguments = {
        "raw_dim": 6,
        "anchor_dim": 13,
        "synaptic_degree": 3,
        "coding_level": 0.25,
        "anchor_ridge": 0.2,
        "residual_ridge": 0.4,
        "complement_ridge": 0.3,
        "energy_threshold": 0.9,
        "max_rank": 4,
        "min_rank": 1,
        "minimum_objective_gain": 0.0,
        "seed": 2025,
        "dtype": DTYPE,
    }
    arguments.update(overrides)
    return CARSFLYLearner(**arguments)


def stream():
    features = torch.randn(48, 6, generator=torch.Generator().manual_seed(81))
    labels = torch.tensor([20, 4, 12, 8] * 12)
    return features, labels


def test_streaming_learner_is_global_finite_and_task_id_free():
    model = learner()
    features, labels = stream()
    model.update(features[:17], labels[:17])
    model.update(features[17:31], labels[17:31])
    model.update(features[31:], labels[31:])
    logits = model.predict_logits(features[:9])

    assert logits.shape == (9, 4)
    assert bool(torch.isfinite(logits).all())
    assert model.class_ids == [4, 8, 12, 20]
    assert "task_id" not in inspect.signature(model.predict_logits).parameters
    assert 1 <= model.diagnostics["effective_rank"] <= model.max_rank
    assert model.diagnostics["geometry"] == "adaptive_conditional_schur"
    model.assert_exemplar_free_state()


def test_streaming_statistics_equal_single_batch_learner():
    streamed = learner()
    batch = learner()
    features, labels = stream()
    streamed.update(features[:11], labels[:11])
    streamed.update(features[11:37], labels[11:37])
    streamed.update(features[37:], labels[37:])
    batch.update(features, labels)

    for name in ("G_pp", "G_xx", "H_px", "Q_p", "Q_x", "counts"):
        torch.testing.assert_close(
            getattr(streamed.statistics, name),
            getattr(batch.statistics, name),
            atol=1e-10,
            rtol=1e-10,
        )
    torch.testing.assert_close(
        streamed.predict_logits(features[:8]),
        batch.predict_logits(features[:8]),
        atol=1e-9,
        rtol=1e-9,
    )


def test_full_energy_rank_matches_full_raw_residual():
    cars = learner(energy_threshold=1.0, max_rank=6)
    full = CRTSOHOLearner(
        method="full_raw_residual",
        raw_dim=6,
        anchor_dim=13,
        synaptic_degree=3,
        coding_level=0.25,
        anchor_ridge=0.2,
        residual_ridge=0.4,
        complement_ridge=0.3,
        requested_rank=6,
        seed=2025,
        dtype=DTYPE,
        anchor_projection=cars.anchor.projection_matrix,
    )
    features, labels = stream()
    cars.update(features, labels)
    full.update(features, labels)

    torch.testing.assert_close(
        cars.predict_logits(features),
        full.predict_logits(features),
        atol=1e-9,
        rtol=1e-9,
    )
    assert cars.diagnostics["retained_correction_energy"] == pytest.approx(
        1.0, abs=1e-12
    )


def test_checkpoint_contains_only_configuration_projection_and_statistics():
    original = learner()
    features, labels = stream()
    original.update(features, labels)
    expected = original.predict_logits(features[:7])
    state = original.state_dict()

    assert set(state) == {
        "version",
        "method",
        "raw_dim",
        "anchor_dim",
        "synaptic_degree",
        "coding_level",
        "anchor_ridge",
        "residual_ridge",
        "complement_ridge",
        "energy_threshold",
        "max_rank",
        "min_rank",
        "minimum_objective_gain",
        "seed",
        "anchor_projection",
        "statistics",
    }
    assert not any(
        token in str(state).lower()
        for token in ("sample", "historical", "replay", "feature_cache")
    )
    restored = learner()
    restored.load_state_dict(state)
    torch.testing.assert_close(
        restored.predict_logits(features[:7]), expected, atol=1e-9, rtol=1e-9
    )


def test_checkpoint_configuration_mismatch_fails_closed():
    original = learner()
    features, labels = stream()
    original.update(features, labels)
    incompatible = learner(energy_threshold=0.8)
    with pytest.raises(ValueError, match="energy_threshold"):
        incompatible.load_state_dict(original.state_dict())


def test_minimum_gain_trigger_can_keep_anchor_only():
    model = learner(minimum_objective_gain=1e30)
    features, labels = stream()
    model.update(features, labels)
    assert model.diagnostics["effective_rank"] == 0
    assert model.directions is None
    assert model.residual_classifier is None
    assert bool(torch.isfinite(model.predict_logits(features[:5])).all())
