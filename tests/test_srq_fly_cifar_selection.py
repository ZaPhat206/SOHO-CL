import argparse
import json
from pathlib import Path

import pytest
import torch

from methods.srq_fly.storage import projected_srq_state_bytes
from tools import srq_fly_cifar_selection as d5
from tools.srq_fly_d2_state_match import exact_fly_state_bytes


def _config(tmp_path: Path) -> Path:
    feature_dim, classes, large_dim, matched_dim = 7, 6, 18, 13
    storage = {"block_size": 4, "group_size": 4}
    projected = projected_srq_state_bytes(
        feature_dim=feature_dim, expand_dim=large_dim, synaptic_degree=3,
        num_classes=classes, **storage,
    )
    payload = {
        "schema_version": 1, "study_id": "synthetic-cifar-d5",
        "dataset": "CIFAR-100", "dataset_version": "synthetic-cifar",
        "model_name": "tiny", "feature_dim": feature_dim,
        "checkpoint_sha256": "synthetic", "seed": 2025,
        "num_classes": classes, "num_tasks": 3,
        "train_samples": 60, "test_samples": 12,
        "outer_validation_fraction": 0.2, "inner_validation_fraction": 0.2,
        "statistics_dtype": "float32", "solver_dtype": "float32",
        "raw_statistics_dtype": "float64", "selection_lambdas": [1.0, 10.0],
        "fixed_raw_ridge_lambda": 0.1,
        "large_representation": {
            "expand_dim": large_dim, "synaptic_degree": 3,
            "coding_level": 1 / 3, "encode_batch_size": 16,
            "evaluation_batch_size": 16,
        },
        "matched_representation": {
            "expand_dim": matched_dim, "synaptic_degree": 3,
            "coding_level": 1 / 3, "encode_batch_size": 16,
            "evaluation_batch_size": 16,
        },
        "storage": storage,
        "expected_state": {
            "nominal_large_projection_entries": large_dim * 3,
            "nominal_matched_projection_entries": matched_dim * 3,
            "maximum_missing_projection_entries": 1,
            "nominal_exact_large_bytes": projected["exact_fly_total_bytes"],
            "nominal_srq_large_bytes": projected["compressed_total_bytes"],
            "nominal_exact_matched_bytes": exact_fly_state_bytes(
                feature_dim=feature_dim, expand_dim=matched_dim,
                synaptic_degree=3, num_classes=classes,
            ),
        },
        "gates": {
            "maximum_inner_solver_relative_residual": 0.001,
            "maximum_outer_solver_relative_residual": 0.001,
            "maximum_average_gap_to_exact_large_pp": 100.0,
            "maximum_final_gap_to_exact_large_pp": 100.0,
            "minimum_prediction_agreement": 0.001,
            "maximum_state_fraction_of_exact_large": 1.0,
            "maximum_state_mismatch_fraction": 1.0,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _cache(tmp_path: Path) -> Path:
    cache = tmp_path / "features"
    cache.mkdir()
    generator = torch.Generator().manual_seed(91)
    features, labels = [], []
    for class_id in range(6):
        center = torch.randn(7, generator=generator) + class_id
        features.append(center + 0.05 * torch.randn(10, 7, generator=generator))
        labels.append(torch.full((10,), class_id, dtype=torch.long))
    torch.save({"features": torch.cat(features), "labels": torch.cat(labels)}, cache / "train.pt")
    (cache / "metadata.json").write_text(json.dumps({
        "schema_version": 1, "dataset": "CIFAR-100",
        "dataset_version": "synthetic-cifar", "backbone_model": "tiny",
        "checkpoint_sha256": "synthetic", "feature_dim": 7,
        "split_sizes": {"train": 60, "test": 12}, "finite": True,
        "test_features_materialized": False,
    }), encoding="utf-8")
    return cache


def _args(tmp_path: Path):
    return argparse.Namespace(
        config=str(_config(tmp_path)), feature_cache_dir=str(_cache(tmp_path)),
        large_code_cache_dir=str(tmp_path / "large"),
        matched_code_cache_dir=str(tmp_path / "matched"),
        output_dir=str(tmp_path / "output"), device="cpu",
    )


def test_repository_config_locks_closest_cifar_state_match():
    config = d5._read_config(
        Path("configs/srq_fly_cifar100_d5_train_only.json")
    )
    assert config["matched_representation"]["expand_dim"] == 4409
    assert config["expected_state"]["nominal_srq_large_bytes"] == 97166240
    assert config["fixed_raw_ridge_lambda"] == 0.01


def test_d5_is_train_only_complete_and_resumable_on_synthetic_data(tmp_path):
    arguments = _args(tmp_path)
    first = d5.run(arguments)
    second = d5.run(arguments)

    assert first["uses_test_set"] is False
    assert first["held_out_test_authorized"] is False
    assert first["selected_fly_and_srq_lambda"] in (1.0, 10.0)
    assert first["srq_fly_10000"] == second["srq_fly_10000"]
    assert first["gates"]["heldout_test_remained_hidden"] is True
    assert (tmp_path / "output" / "d5_results.json").is_file()


def test_d5_refuses_visible_test_before_wta_cache(tmp_path):
    arguments = _args(tmp_path)
    torch.save({"features": torch.zeros(1, 7)}, Path(arguments.feature_cache_dir) / "test.pt")

    with pytest.raises(RuntimeError, match="refuses a visible test.pt"):
        d5.run(arguments)
    assert not Path(arguments.large_code_cache_dir).exists()


def test_d5_rejects_non_closest_state_match(tmp_path):
    path = _config(tmp_path)
    payload = json.loads(path.read_text())
    payload["matched_representation"]["expand_dim"] = 12
    payload["expected_state"]["nominal_matched_projection_entries"] = 36
    payload["expected_state"]["nominal_exact_matched_bytes"] = exact_fly_state_bytes(
        feature_dim=7, expand_dim=12, synaptic_degree=3, num_classes=6
    )
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="closest non-exceeding"):
        d5._read_config(path)
