import inspect

import pytest
import torch

from methods.mars_soho import MARSExactReplayOracle, MARSSOHOLearner


def _configuration(mode="support_aware"):
    return {
        "feature_dim": 5,
        "expand_dim": 20,
        "density": 0.4,
        "olda_dim": 5,
        "use_etf": True,
        "coding_level": 0.25,
        "ridge_lambda": 0.5,
        "model_mode": mode,
        "pseudo_per_class": 8,
        "pilot_per_class": 6,
        "covariance_rank": 3,
        "shrinkage": 0.25,
        "minimum_per_class": 2,
        "seed": 2025,
    }


def _stream():
    torch.manual_seed(19)
    labels_a = torch.tensor([0] * 7 + [1] * 7)
    labels_b = torch.tensor([2] * 7 + [3] * 7)
    features_a = torch.randn(14, 5, dtype=torch.float64)
    features_b = torch.randn(14, 5, dtype=torch.float64)
    features_a += torch.nn.functional.one_hot(labels_a, 5).double()
    features_b += torch.nn.functional.one_hot(labels_b, 5).double()
    return features_a, labels_a, features_b, labels_b


def test_first_task_matches_exact_replay_oracle_before_approximation_is_needed():
    features_a, labels_a, _, _ = _stream()
    learner = MARSSOHOLearner(**_configuration("shared_gaussian"))
    oracle_config = _configuration()
    for field in (
        "model_mode", "pseudo_per_class", "pilot_per_class", "covariance_rank",
        "shrinkage", "minimum_per_class",
    ):
        oracle_config.pop(field)
    oracle = MARSExactReplayOracle(**oracle_config)
    learner.update(features_a, labels_a)
    oracle.update(features_a, labels_a)
    assert torch.allclose(learner.G, oracle.G, atol=1e-10, rtol=1e-10)
    assert torch.allclose(learner.Q, oracle.Q, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        learner.predict_logits(features_a), oracle.predict_logits(features_a),
        atol=1e-9, rtol=1e-9,
    )


@pytest.mark.parametrize(
    "mode",
    [
        "shared_gaussian",
        "heterogeneous_spherical",
        "support_aware",
        "shuffled_support",
        "turnover_aware",
        "shuffled_turnover",
        "statistic_variance_aware",
        "shuffled_statistic_variance",
    ],
)
def test_all_phase1_modes_run_two_tasks_without_task_id(mode):
    features_a, labels_a, features_b, labels_b = _stream()
    learner = MARSSOHOLearner(**_configuration(mode))
    learner.update(features_a, labels_a)
    learner.update(features_b, labels_b)
    logits = learner.predict_logits(torch.cat((features_a[:2], features_b[:2])))
    assert logits.shape == (4, 4)
    assert bool(torch.isfinite(logits).all())
    assert learner.diagnostics["pseudo_total"] == 16
    assert "task_id" not in inspect.signature(learner.predict_logits).parameters


@pytest.mark.parametrize(
    ("mode", "risk_name"),
    [
        ("turnover_aware", "support_turnover"),
        ("statistic_variance_aware", "statistic_variance"),
    ],
)
def test_phase1b_modes_report_disjoint_pilot_risks(mode, risk_name):
    features_a, labels_a, features_b, labels_b = _stream()
    learner = MARSSOHOLearner(**_configuration(mode))
    learner.update(features_a, labels_a)
    learner.update(features_b, labels_b)
    assert learner.diagnostics["allocation_risk_name"] == risk_name
    assert set(learner.diagnostics["pilot_risks"]) == {
        "certificate_failure", "support_turnover", "statistic_variance"
    }
    values = learner.diagnostics["pilot_risks"][risk_name].values()
    assert all(torch.isfinite(torch.tensor(value)) for value in values)


def test_checkpoint_roundtrip_and_state_inventory_are_exemplar_free():
    features_a, labels_a, features_b, labels_b = _stream()
    learner = MARSSOHOLearner(**_configuration())
    learner.update(features_a, labels_a)
    learner.update(features_b, labels_b)
    state = learner.state_dict()
    serialized_names = repr(state).lower()
    for forbidden in ("feature_history", "label_history", "pseudo_features", "samples"):
        assert forbidden not in serialized_names
    restored = MARSSOHOLearner(**_configuration())
    restored.load_state_dict(state)
    queries = torch.cat((features_a[:3], features_b[:3]))
    assert restored.class_ids == learner.class_ids
    assert torch.allclose(
        restored.predict_logits(queries), learner.predict_logits(queries),
        atol=1e-10, rtol=1e-10,
    )
    restored.assert_exemplar_free_state()


def test_oracle_discloses_feature_replay_and_mars_state_does_not_scale_by_rows():
    features_a, labels_a, features_b, labels_b = _stream()
    learner = MARSSOHOLearner(**_configuration())
    oracle_config = _configuration()
    for field in (
        "model_mode", "pseudo_per_class", "pilot_per_class", "covariance_rank",
        "shrinkage", "minimum_per_class",
    ):
        oracle_config.pop(field)
    oracle = MARSExactReplayOracle(**oracle_config)
    for features, labels in ((features_a, labels_a), (features_b, labels_b)):
        learner.update(features, labels)
        oracle.update(features, labels)
    assert learner.is_exemplar_free is True
    assert oracle.is_exemplar_free is False
    assert sum(len(value) for value in oracle.feature_history) == 28
    assert not any(28 in tensor.shape for tensor in learner.persistent_tensors().values())
