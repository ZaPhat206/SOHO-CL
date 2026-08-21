"""Synthetic integration tests for the locked SRQ-FLY D1 runner."""

import argparse
import json
from pathlib import Path

import pytest
import torch

from tools import srq_fly_d1


def _config():
    return {
        "schema_version": 1, "study_id": "srq-fly-synthetic-d1",
        "dataset": "Synthetic", "model_name": "tiny_backbone",
        "checkpoint_sha256": "synthetic-checkpoint", "seed": 2025,
        "num_classes": 6, "num_tasks": 3, "diagnostic_tasks": 3,
        "validation_fraction": 0.25, "statistics_dtype": "float64",
        "solver_dtype": "float64", "ridge_lambda": 100.0,
        "raw_ridge_lambda": 1.0,
        "large_representation": {
            "expand_dim": 18, "synaptic_degree": 3, "coding_level": 1 / 3,
            "encode_batch_size": 16, "evaluation_batch_size": 16,
        },
        "compact_representation": {
            "expand_dim": 12, "synaptic_degree": 3, "coding_level": 1 / 3,
            "encode_batch_size": 16, "evaluation_batch_size": 16,
        },
        "storage": {"block_size": 7, "group_size": 5},
        "gates": {
            "maximum_solver_relative_residual": 1e-8,
            "maximum_average_gap_to_exact_fly_pp": 100.0,
            "maximum_final_gap_to_exact_fly_pp": 100.0,
            "maximum_state_fraction_of_exact_fly": 1.0,
            "minimum_prediction_agreement": 0.0,
            "minimum_gain_over_direct_int8_pp": 0.0,
            "maximum_float16_gap_to_exact_fly_pp": 100.0,
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
    }), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    return cache, config_path


def _args(tmp_path, cache, config_path):
    return argparse.Namespace(
        config=str(config_path), feature_cache_dir=str(cache),
        large_code_cache_dir=str(tmp_path / "wta_large"),
        compact_code_cache_dir=str(tmp_path / "wta_compact"),
        output_dir=str(tmp_path / "output"), device="cpu", require_test_hidden=True,
    )


def test_d1_runner_is_paired_train_only_and_resumable(tmp_path):
    cache, config_path = _fixture(tmp_path)
    arguments = _args(tmp_path, cache, config_path)
    first = srq_fly_d1.run(arguments)
    second = srq_fly_d1.run(arguments)
    assert first["uses_test_set"] is False and first["held_out_test_authorized"] is False
    assert first["diagnostic_tasks"] == 3
    assert first["results"] == second["results"]
    assert len(first["paired_diagnostics"]) == 3
    exact = next(item for item in first["results"] if item["method"] == "exact_fly_10000")
    srq = next(item for item in first["results"] if item["method"] == "srq_int8")
    assert exact["stage_accuracy"] == [row["exact_accuracy"] for row in first["paired_diagnostics"]]
    assert srq["stage_accuracy"] == [row["approximate_accuracy"] for row in first["paired_diagnostics"]]
    assert all(0 <= row["prediction_agreement"] <= 1 for row in first["paired_diagnostics"])
    assert all(row["relative_logit_frobenius_error"] >= 0 for row in first["paired_diagnostics"])
    assert {result["method"] for result in first["results"]} == {
        "exact_fly_10000", "srq_int8", "exact_fly_4096", "raw_ridge",
        "direct_int8_gram", "sqrt_float16",
    }
    serialized = json.dumps(first)
    assert '"predictions"' not in serialized and '"validation_features"' not in serialized
    assert (tmp_path / "output" / "d1_results.json").is_file()


def test_d1_refuses_visible_test_cache(tmp_path):
    cache, config_path = _fixture(tmp_path)
    torch.save({"features": torch.zeros(1, 7)}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="held-out test.pt is visible"):
        srq_fly_d1.run(_args(tmp_path, cache, config_path))


def test_d1_config_is_strict_full_stream_and_seed_2025(tmp_path):
    config = _config()
    config["diagnostic_tasks"] = 2
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="complete valid task stream"):
        srq_fly_d1._read_config(path)
    config = _config()
    config["seed"] = 1993
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="seed 2025"):
        srq_fly_d1._read_config(path)
