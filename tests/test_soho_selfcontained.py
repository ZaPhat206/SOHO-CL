import json
from pathlib import Path

import pytest
import torch

from methods.cached_replay_baselines import CachedFlyCLFidelity, CachedSOHOReplayFidelity
from tools import soho_selfcontained as runner


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/soho_selfcontained_final.json"


def test_locked_protocol_and_method_sources_match():
    protocol = runner._read_protocol(PROTOCOL)
    assert runner._verify_method_identity(protocol) == protocol["method_identity"]
    assert protocol["final_evaluation"]["methods"] == list(runner.METHODS)
    assert len(runner._soho_candidates(protocol)) == 35


def test_two_stage_ranking_and_sparse_near_tie_are_predeclared():
    selection = {
        "anchor_density": 0.3, "anchor_coding_level": 0.3,
        "top_density_count": 2, "top_coding_count": 3,
    }
    stage1 = [
        {"config": {"density": density, "coding_level": 0.3}, "mean_inner_aia": score}
        for density, score in ((0.1, 80.0), (0.2, 82.0), (0.3, 81.0))
    ] + [
        {"config": {"density": 0.3, "coding_level": coding}, "mean_inner_aia": score}
        for coding, score in ((0.1, 79.0), (0.2, 81.5), (0.4, 82.5))
    ]
    density, coding = runner._rank_stage1_sensitivity(stage1, selection)
    assert [item["config"]["density"] for item in density[:2]] == [0.2, 0.3]
    assert [item["config"]["coding_level"] for item in coding[:3]] == [0.4, 0.2, 0.3]
    interactions = [
        {"valid": True, "mean_inner_aia": 90.00,
         "config": {"density": 0.2, "coding_level": 0.2}},
        {"valid": True, "mean_inner_aia": 90.04,
         "config": {"density": 0.3, "coding_level": 0.3}},
        {"valid": True, "mean_inner_aia": 89.00,
         "config": {"density": 0.1, "coding_level": 0.1}},
    ]
    selected, best = runner._select_sparse_near_tie(interactions, tolerance_pp=0.05)
    assert best == pytest.approx(90.04)
    assert selected["config"] == {"density": 0.2, "coding_level": 0.2}


def test_nested_split_is_deterministic_stratified_and_disjoint():
    labels = torch.arange(4).repeat_interleave(25)
    order = [2, 0, 3, 1]
    first = runner._nested_parts(labels, order, tasks=2, split_seed=2025,
                                 outer_fraction=0.2, inner_fraction=0.2)
    second = runner._nested_parts(labels, order, tasks=2, split_seed=2025,
                                  outer_fraction=0.2, inner_fraction=0.2)
    for left_group, right_group in zip(first, second):
        assert all(torch.equal(left, right) for left, right in zip(left_group, right_group))
    inner_fit, inner_validation, outer_fit, outer_validation = first
    for task in range(2):
        assert not set(inner_fit[task].tolist()) & set(inner_validation[task].tolist())
        assert not set(outer_fit[task].tolist()) & set(outer_validation[task].tolist())
        assert set(inner_fit[task].tolist()) | set(inner_validation[task].tolist()) == set(outer_fit[task].tolist())


def test_cache_validation_fails_closed_when_test_is_visible(tmp_path):
    protocol = {
        "backbone": {
            "model_name": "vit_base_patch16_224", "checkpoint_sha256": "hash",
            "preprocessing": "vit", "feature_dim": 3,
        },
        "datasets": {"cifar100": {"dataset": "CIFAR-100", "train_samples": 4,
                                      "test_samples": 2, "num_classes": 2}},
    }
    (tmp_path / "metadata.json").write_text(json.dumps({
        "dataset": "CIFAR-100", "backbone_model": "vit_base_patch16_224",
        "checkpoint_sha256": "hash", "preprocessing": "vit",
    }))
    torch.save({"features": torch.randn(4, 3), "labels": torch.tensor([0, 0, 1, 1])}, tmp_path / "train.pt")
    torch.save({"features": torch.randn(2, 3), "labels": torch.tensor([0, 1])}, tmp_path / "test.pt")
    with pytest.raises(RuntimeError, match="refuses a visible test.pt"):
        runner._validate_cache(tmp_path, protocol, "cifar100", require_test=False)


def test_soho_state_audit_discloses_and_counts_feature_replay():
    torch.manual_seed(7)
    learner = CachedSOHOReplayFidelity(
        feature_dim=4, expand_dim=12, density=0.5, olda_dim=4, use_etf=True,
        coding_level=0.25, num_classes=4, ridge_lower=-1, ridge_upper=2,
        seed=11, replay_chunk_size=8, gcv_sample_size=8,
    )
    features = torch.randn(8, 4)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    learner.update(features, labels)
    audit = runner._state_audit(learner, "soho_replay_fidelity", retained_rows=8)
    assert audit["exemplar_free"] is False
    assert audit["historical_feature_rows"] == 8
    assert audit["historical_label_rows"] == 8
    assert learner.state_dict()["feature_history"][0].shape == (8, 4)


def test_fly_state_audit_remains_sample_free():
    torch.manual_seed(9)
    learner = CachedFlyCLFidelity(
        feature_dim=4, expand_dim=16, synaptic_degree=3, coding_level=0.25,
        num_classes=4, ridge_lower=-1, ridge_upper=2, seed=13,
    )
    features = torch.randn(8, 4)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    learner.update(features, labels)
    audit = runner._state_audit(learner, "flycl_fidelity", retained_rows=8)
    assert audit["exemplar_free"] is True
    assert audit["historical_feature_rows"] == 0


def test_raw_ridge_streaming_matches_batch_oracle():
    torch.manual_seed(17)
    x1, x2 = torch.randn(9, 5), torch.randn(7, 5)
    y1 = torch.tensor([0, 1, 2] * 3)
    y2 = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    learner = runner._RawRidge(5, 3, ridge=0.1, device="cpu")
    learner.update(x1, y1); learner.update(x2, y2)
    x, y = torch.cat((x1, x2)).double(), torch.cat((y1, y2))
    targets = torch.nn.functional.one_hot(y, 3).double()
    expected = torch.linalg.solve(x.T @ x + 0.1 * torch.eye(5, dtype=torch.float64), x.T @ targets)
    assert torch.allclose(learner.G, x.T @ x, atol=1e-10, rtol=1e-10)
    assert torch.allclose(learner.Q, x.T @ targets, atol=1e-10, rtol=1e-10)
    assert torch.allclose(learner.weights, expected, atol=1e-10, rtol=1e-10)


def test_metric_definition_and_soho_minus_baseline_sign():
    matrix = [[80.0], [70.0, 90.0], [60.0, 85.0, 95.0]]
    metrics = runner._metrics(matrix)
    assert metrics["final_accuracy"] == pytest.approx(80.0)
    assert metrics["average_incremental_accuracy"] == pytest.approx((80.0 + 80.0 + 80.0) / 3)
    assert metrics["forgetting"] == pytest.approx((20.0 + 5.0) / 2)


def test_toy_stream_runs_all_final_methods_without_task_id_at_inference():
    torch.manual_seed(23)
    labels = torch.tensor([0] * 6 + [1] * 6 + [2] * 6 + [3] * 6)
    features = torch.randn(len(labels), 4) + torch.nn.functional.one_hot(labels, 4).float()
    stream = {"features": features, "labels": labels}
    training_parts = [torch.tensor(list(range(0, 4)) + list(range(6, 10))),
                      torch.tensor(list(range(12, 16)) + list(range(18, 22)))]
    evaluation_parts = [torch.tensor([4, 5, 10, 11]), torch.tensor([16, 17, 22, 23])]
    protocol = {
        "backbone": {"feature_dim": 4},
        "soho_fixed": {"expand_dim": 12, "olda_dim": 4, "ridge_lower": -1,
                       "ridge_upper": 2, "replay_chunk_size": 8, "gcv_sample_size": 8},
        "fly_fixed": {"expand_dim": 16, "synaptic_degree": 3, "coding_level": 0.25,
                      "ridge_lower": -1, "ridge_upper": 2},
    }
    dataset = {"num_classes": 4}
    soho = {"density": 0.5, "coding_level": 0.25, "use_etf": True}
    for method in runner.METHODS:
        result = runner._evaluate(
            method, protocol, dataset, seed=31, stream=stream,
            training_parts=training_parts, evaluation_parts=evaluation_parts,
            soho_config=soho, raw_ridge=0.1, device_name="cpu", uses_test_set=False,
        )
        assert result["status"] == "complete"
        assert len(result["stage_accuracy"]) == 2
        assert result["state_audit"]["exemplar_free"] is (method != "soho_replay_fidelity")
