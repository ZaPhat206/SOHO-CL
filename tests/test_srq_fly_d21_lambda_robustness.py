"""Synthetic tests for nested SRQ-FLY D2.1 lambda robustness."""

import argparse
import json
import random
from pathlib import Path

import pytest
import torch

from tools import srq_fly_d21_lambda_robustness as d21
from tools.experiment_runner import split, train_validation_indices
from tools.twa_fly_pilot import _sequence_sha256, _sha256_file


def _fixture(tmp_path: Path):
    cache = tmp_path / "features"
    cache.mkdir()
    generator = torch.Generator().manual_seed(71)
    features, labels = [], []
    for class_id in range(6):
        center = torch.randn(7, generator=generator) + 0.5 * class_id
        features.append(center + 0.2 * torch.randn(12, 7, generator=generator))
        labels.append(torch.full((12,), class_id, dtype=torch.long))
    train = {"features": torch.cat(features), "labels": torch.cat(labels)}
    torch.save(train, cache / "train.pt")
    (cache / "metadata.json").write_text(json.dumps({
        "schema_version": 1, "dataset": "Synthetic", "backbone_model": "tiny_backbone",
        "checkpoint_sha256": "synthetic-checkpoint", "feature_dim": 7,
        "finite": True, "test_features_materialized": False,
    }), encoding="utf-8")
    class_order = random.Random(2025).sample(list(range(6)), 6)
    tasks = split(train["labels"], class_order, 3)
    outer_training, outer_validation = train_validation_indices(
        train["labels"], tasks, 2025, 0.25
    )
    train_hash = _sha256_file(cache / "train.pt")
    d2_path = tmp_path / "d2_results.json"
    d2_payload = {
        "status": "PASS_REVIEW_D2", "uses_test_set": False,
        "held_out_test_authorized": False, "class_order": class_order,
        "provenance": {
            "config_sha256": "synthetic-d2-config",
            "runner_git_commit": "synthetic-d2-commit", "runner_git_dirty": False,
            "train_sha256": train_hash,
            "training_indices_sha256": _sequence_sha256(outer_training),
            "validation_indices_sha256": _sequence_sha256(outer_validation),
        },
        "gates": {"all": True},
        "state_matched_exact_fly": {
            "method": "exact_fly_18", "status": "complete", "uses_test_set": False,
            "exemplar_free": True, "validation_average_accuracy": 90.0,
            "stage_accuracy": [90.0, 90.0, 90.0], "persistent_state_bytes": 2896,
        },
        "d1_reference": {
            "srq_validation_average_accuracy": 100.0,
            "srq_final_accuracy": 100.0, "srq_persistent_state_bytes": 2900,
        },
    }
    d2_path.write_text(json.dumps(d2_payload), encoding="utf-8")
    reference = {
        "result_sha256": _sha256_file(d2_path),
        "config_sha256": "synthetic-d2-config",
        "runner_git_commit": "synthetic-d2-commit", "train_sha256": train_hash,
        "outer_training_indices_sha256": _sequence_sha256(outer_training),
        "outer_validation_indices_sha256": _sequence_sha256(outer_validation),
        "srq_persistent_state_bytes": 2900,
        "srq_validation_average_accuracy": 100.0, "srq_final_accuracy": 100.0,
        "d2_exact_validation_average_accuracy": 90.0,
        "d2_exact_final_accuracy": 90.0,
        "d2_exact_persistent_state_bytes": 2896,
    }
    config = {
        "schema_version": 1, "study_id": "srq-fly-synthetic-d21",
        "dataset": "Synthetic", "model_name": "tiny_backbone", "feature_dim": 7,
        "checkpoint_sha256": "synthetic-checkpoint", "seed": 2025,
        "num_classes": 6, "num_tasks": 3, "outer_validation_fraction": 0.25,
        "inner_validation_fraction": 0.25, "statistics_dtype": "float32",
        "selection_lambdas": [100.0, 1_000_000.0],
        "representation": {
            "expand_dim": 18, "synaptic_degree": 3, "coding_level": 1 / 3,
            "encode_batch_size": 16, "evaluation_batch_size": 16,
        },
        "reference_d2": reference, "expected_exact_state_bytes": 2896,
        "gates": {
            "maximum_solver_relative_residual": 1e-4,
            "maximum_state_mismatch_fraction": 0.99,
            "minimum_srq_average_gain_pp": 0.0,
            "minimum_srq_final_gain_pp": 0.0,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return cache, config_path, d2_path, outer_training, outer_validation


def _args(tmp_path, cache, config_path, d2_path):
    return argparse.Namespace(
        config=str(config_path), feature_cache_dir=str(cache),
        code_cache_dir=str(tmp_path / "wta"), d2_result=str(d2_path),
        output_dir=str(tmp_path / "output"), device="cpu", require_test_hidden=True,
    )


def test_nested_split_is_disjoint_and_covers_only_outer_training(tmp_path):
    cache, _, _, outer_training, outer_validation = _fixture(tmp_path)
    train = torch.load(cache / "train.pt", weights_only=True)
    inner_fit, inner_validation = train_validation_indices(
        train["labels"], outer_training, 2025, 0.25
    )
    d21._validate_nested_parts(
        outer_training, outer_validation, inner_fit, inner_validation
    )
    for outer_fit, outer_val, fit, val in zip(
        outer_training, outer_validation, inner_fit, inner_validation
    ):
        assert set(fit.tolist()).isdisjoint(val.tolist())
        assert set(fit.tolist()) | set(val.tolist()) == set(outer_fit.tolist())
        assert (set(fit.tolist()) | set(val.tolist())).isdisjoint(outer_val.tolist())


def test_candidate_tie_break_is_smaller_lambda():
    selected = d21._choose_candidate([
        {"ridge_lambda": 1_000_000.0, "validation_average_accuracy": 88.0},
        {"ridge_lambda": 100.0, "validation_average_accuracy": 88.0},
    ])
    assert selected["ridge_lambda"] == 100.0


def test_exact_result_contract_rejects_sample_state_or_wrong_average():
    result = {
        "method": "candidate", "status": "complete", "uses_test_set": False,
        "exemplar_free": True, "ridge_lambda": 100.0,
        "persistent_state_bytes": 2896, "stage_accuracy": [80.0, 90.0],
        "validation_average_accuracy": 85.0,
        "maximum_solver_relative_residual": 1e-7,
    }
    d21._validate_exact_result(
        result, name="candidate", ridge_lambda=100.0, num_tasks=2,
        expected_state_bytes=2896,
    )
    result["validation_average_accuracy"] = 84.0
    with pytest.raises(ValueError, match="invalid exact-FLY metrics"):
        d21._validate_exact_result(
            result, name="candidate", ridge_lambda=100.0, num_tasks=2,
            expected_state_bytes=2896,
        )
    result["validation_average_accuracy"] = 85.0
    result["sample_features"] = torch.zeros(3, 7)
    with pytest.raises(ValueError, match="invalid exact-FLY result contract"):
        d21._validate_exact_result(
            result, name="candidate", ridge_lambda=100.0, num_tasks=2,
            expected_state_bytes=2896,
        )


def test_d21_runner_is_nested_train_only_and_resumable(tmp_path):
    cache, config_path, d2_path, _, _ = _fixture(tmp_path)
    arguments = _args(tmp_path, cache, config_path, d2_path)
    first = d21.run(arguments)
    second = d21.run(arguments)
    assert first == second
    assert first["uses_test_set"] is False
    assert first["uses_outer_validation_for_selection"] is False
    assert first["selection"]["candidate_count"] == 2
    assert first["selection"]["selected_lambda"] in {100.0, 1_000_000.0}
    assert first["tuned_state_matched_exact_fly"]["persistent_state_bytes"] == 2896
    assert first["gates"]["outer_validation_not_used_for_selection"] is True
    selection = json.loads(
        (tmp_path / "output" / "lambda_selection.json").read_text()
    )
    assert selection["uses_outer_validation_for_selection"] is False
    assert "outer_validation_indices" not in selection
    assert (tmp_path / "output" / "d21_results.json").is_file()


def test_d21_refuses_visible_test_and_tampered_d2(tmp_path):
    cache, config_path, d2_path, _, _ = _fixture(tmp_path)
    torch.save({"features": torch.zeros(1, 7)}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="held-out test.pt is visible"):
        d21.run(_args(tmp_path, cache, config_path, d2_path))
    (cache / "test.pt").unlink()
    payload = json.loads(d2_path.read_text())
    payload["d1_reference"]["srq_final_accuracy"] = 99.0
    d2_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="D2 result SHA-256 mismatch"):
        d21.run(_args(tmp_path, cache, config_path, d2_path))


def test_d21_config_rejects_unlocked_grid_and_duplicate_keys(tmp_path):
    _, config_path, _, _, _ = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["selection_lambdas"] = [100.0, 10.0]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="selection_lambdas"):
        d21._read_config(config_path)
    config_path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate config key"):
        d21._read_config(config_path)
