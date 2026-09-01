"""Synthetic gates for the isolated Priority-1 SRQ-FLY ablation."""

import argparse
import json
from pathlib import Path

import pytest
import torch

from tools import srq_fly_priority1_ablation as priority1


def _config():
    return {
        "schema_version": 1,
        "study_id": "priority1-synthetic",
        "dataset": "Synthetic",
        "model_name": "tiny_backbone",
        "checkpoint_sha256": "synthetic-checkpoint",
        "feature_dim": 7,
        "seed": 2025,
        "num_classes": 6,
        "num_tasks": 3,
        "validation_fraction": 0.25,
        "statistics_dtype": "float32",
        "solver_dtype": "float32",
        "fly_ridge_lambda": 100.0,
        "raw_ridge_lambda": 1.0,
        "srq_update_backend": "gram_cholesky_direct",
        "large_representation": {
            "expand_dim": 18, "synaptic_degree": 3, "coding_level": 1 / 3,
            "encode_batch_size": 16, "evaluation_batch_size": 16,
        },
        "state_matched_representation": {
            "expand_dim": 8, "synaptic_degree": 3, "coding_level": 1 / 3,
            "encode_batch_size": 16, "evaluation_batch_size": 16,
        },
        "storage": {"block_size": 7, "group_size": 5},
        "gates": {
            "maximum_solver_relative_residual": 1e-3,
            "maximum_srq_gap_to_exact_fly_pp": 100.0,
            "maximum_float16_gap_to_exact_fly_pp": 100.0,
            "maximum_state_match_error_fraction": 100.0,
            "maximum_srq_state_fraction_of_exact": 100.0,
            "minimum_system_update_speedup": 0.0,
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
    torch.save({"features": torch.cat(features), "labels": torch.cat(labels)}, cache / "train.pt")
    (cache / "metadata.json").write_text(json.dumps({
        "schema_version": 1, "dataset": "Synthetic", "backbone_model": "tiny_backbone",
        "checkpoint_sha256": "synthetic-checkpoint", "feature_dim": 7,
        "finite": True, "test_features_materialized": False,
    }))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config()))
    return cache, config


def test_priority1_worker_is_train_only_and_reports_measurement_scope(tmp_path):
    cache, config = _fixture(tmp_path)
    output = tmp_path / "worker.json"
    result = priority1.run_worker(argparse.Namespace(
        config=str(config), feature_cache_dir=str(cache),
        large_code_cache_dir=str(tmp_path / "large"),
        matched_code_cache_dir=str(tmp_path / "matched"),
        method="srq_int8_optimized", output=str(output), device="cpu",
    ))
    assert result["status"] == "complete" and result["uses_test_set"] is False
    assert result["peak_cuda_allocated_bytes"] is None
    assert result["total_update_seconds"] > 0
    assert output.is_file() and not (cache / "test.pt").exists()


def test_priority1_refuses_visible_test_and_wrong_backend(tmp_path):
    cache, config = _fixture(tmp_path)
    torch.save({}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="test.pt"):
        priority1.run_worker(argparse.Namespace(
            config=str(config), feature_cache_dir=str(cache),
            large_code_cache_dir=str(tmp_path / "large"),
            matched_code_cache_dir=str(tmp_path / "matched"),
            method="raw_ridge", output=str(tmp_path / "out.json"), device="cpu",
        ))
    payload = _config()
    payload["srq_update_backend"] = "stacked_qr"
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="backend identity"):
        priority1._read_config(config)
