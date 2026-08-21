"""Synthetic tests for the SRQ-FLY D2 state-matched control."""

import argparse
import json
import random
from pathlib import Path

import pytest
import torch

from models.flyhash import FlyHash
from tools import srq_fly_d2_state_match as d2
from tools.experiment_runner import split, train_validation_indices
from tools.twa_fly_pilot import _sequence_sha256, _sha256_file


def _base_config(reference):
    return {
        "schema_version": 1, "study_id": "srq-fly-synthetic-d2",
        "dataset": "Synthetic", "model_name": "tiny_backbone", "feature_dim": 7,
        "checkpoint_sha256": "synthetic-checkpoint", "seed": 2025,
        "num_classes": 6, "num_tasks": 3, "validation_fraction": 0.25,
        "statistics_dtype": "float32", "ridge_lambda": 100.0,
        "representation": {
            "expand_dim": 18, "synaptic_degree": 3, "coding_level": 1 / 3,
            "encode_batch_size": 16, "evaluation_batch_size": 16,
        },
        "reference_d1": reference,
        "expected_exact_state_bytes": 2896,
        "gates": {
            "maximum_solver_relative_residual": 1e-4,
            "maximum_state_mismatch_fraction": 0.99,
            "minimum_srq_average_gain_pp": 0.0,
            "minimum_srq_final_gain_pp": 0.0,
        },
    }


def _fixture(tmp_path: Path):
    cache = tmp_path / "features"
    cache.mkdir()
    generator = torch.Generator().manual_seed(47)
    features, labels = [], []
    for class_id in range(6):
        center = torch.randn(7, generator=generator) + class_id / 3
        features.append(center + 0.15 * torch.randn(10, 7, generator=generator))
        labels.append(torch.full((10,), class_id, dtype=torch.long))
    train = {"features": torch.cat(features), "labels": torch.cat(labels)}
    torch.save(train, cache / "train.pt")
    (cache / "metadata.json").write_text(json.dumps({
        "schema_version": 1, "dataset": "Synthetic", "backbone_model": "tiny_backbone",
        "checkpoint_sha256": "synthetic-checkpoint", "feature_dim": 7,
        "finite": True, "test_features_materialized": False,
    }), encoding="utf-8")
    class_order = random.Random(2025).sample(list(range(6)), 6)
    parts = split(train["labels"], class_order, 3)
    training, validation = train_validation_indices(train["labels"], parts, 2025, 0.25)
    reference = {
        "config_sha256": "synthetic-d1-config",
        "runner_git_commit": "synthetic-d1-commit",
        "train_sha256": _sha256_file(cache / "train.pt"),
        "training_indices_sha256": _sequence_sha256(training),
        "validation_indices_sha256": _sequence_sha256(validation),
        "srq_persistent_state_bytes": 2900,
        "srq_validation_average_accuracy": 100.0,
        "srq_final_accuracy": 100.0,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_base_config(reference)), encoding="utf-8")
    d1_path = tmp_path / "d1_results.json"
    d1_path.write_text(json.dumps({
        "status": "STOP_SRQ_FLY_D1", "uses_test_set": False,
        "held_out_test_authorized": False, "diagnostic_tasks": 3,
        "class_order": class_order,
        "provenance": {
            "config_sha256": reference["config_sha256"],
            "runner_git_commit": reference["runner_git_commit"],
            "train_sha256": reference["train_sha256"],
            "training_indices_sha256": reference["training_indices_sha256"],
            "validation_indices_sha256": reference["validation_indices_sha256"],
            "runner_git_dirty": False,
        },
        "results": [{
            "method": "srq_int8", "persistent_state_bytes": 2900,
            "validation_average_accuracy": 100.0,
            "stage_accuracy": [100.0, 100.0, 100.0],
            "status": "complete", "uses_test_set": False, "exemplar_free": True,
        }],
    }), encoding="utf-8")
    return cache, config_path, d1_path


def _args(tmp_path, cache, config_path, d1_path):
    return argparse.Namespace(
        config=str(config_path), feature_cache_dir=str(cache),
        code_cache_dir=str(tmp_path / "wta"), d1_result=str(d1_path),
        output_dir=str(tmp_path / "output"), device="cpu", require_test_hidden=True,
    )


def test_state_formula_selects_closest_non_exceeding_dimension():
    assert d2.exact_fly_state_bytes(
        feature_dim=768, expand_dim=4518, synaptic_degree=300, num_classes=200
    ) == 105149848
    assert d2.exact_fly_state_bytes(
        feature_dim=768, expand_dim=4519, synaptic_degree=300, num_classes=200
    ) == 105191196


def test_smaller_exact_fly_projection_is_seed_matched_prefix():
    with torch.random.fork_rng():
        torch.manual_seed(2025)
        smaller = FlyHash(in_dim=7, expand_dim=18, synaptic_degree=3)
        torch.manual_seed(2025)
        larger = FlyHash(in_dim=7, expand_dim=30, synaptic_degree=3)
    torch.testing.assert_close(
        smaller.projection_matrix, larger.projection_matrix[:18], rtol=0, atol=0
    )


def test_d2_runner_is_train_only_state_matched_and_resumable(tmp_path):
    cache, config_path, d1_path = _fixture(tmp_path)
    arguments = _args(tmp_path, cache, config_path, d1_path)
    first = d2.run(arguments)
    second = d2.run(arguments)
    assert first["uses_test_set"] is False and first["held_out_test_authorized"] is False
    assert first["state_matched_exact_fly"] == second["state_matched_exact_fly"]
    assert first["state_matched_exact_fly"]["persistent_state_bytes"] == 2896
    assert first["gates"]["analytic_state_accounting_matches_runtime"] is True
    assert first["gates"]["heldout_test_remained_hidden"] is True
    assert (tmp_path / "output" / "d2_results.json").is_file()


def test_d2_refuses_visible_test_and_wrong_d1_reference(tmp_path):
    cache, config_path, d1_path = _fixture(tmp_path)
    torch.save({"features": torch.zeros(1, 7)}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="held-out test.pt is visible"):
        d2.run(_args(tmp_path, cache, config_path, d1_path))
    (cache / "test.pt").unlink()
    payload = json.loads(d1_path.read_text())
    payload["results"][0]["persistent_state_bytes"] = 2901
    d1_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="srq_persistent_state_bytes"):
        d2.run(_args(tmp_path, cache, config_path, d1_path))


def test_d2_config_rejects_non_closest_dimension(tmp_path):
    reference = {
        "config_sha256": "x", "runner_git_commit": "y", "train_sha256": "z",
        "training_indices_sha256": "a", "validation_indices_sha256": "b",
        "srq_persistent_state_bytes": 2900,
        "srq_validation_average_accuracy": 1.0, "srq_final_accuracy": 1.0,
    }
    config = _base_config(reference)
    config["representation"]["expand_dim"] = 17
    config["expected_exact_state_bytes"] = d2.exact_fly_state_bytes(
        feature_dim=7, expand_dim=17, synaptic_degree=3, num_classes=6
    )
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="closest non-exceeding"):
        d2._read_config(path)


def test_d2_config_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate config key: schema_version"):
        d2._read_config(path)
