"""Synthetic integration tests for the locked SRQ-FLY D0 runner."""

import argparse
import json
from pathlib import Path

import pytest
import torch

from tools import srq_fly_d0


def _config():
    return {
        "schema_version": 1,
        "study_id": "srq-fly-synthetic-d0",
        "dataset": "Synthetic",
        "model_name": "tiny_backbone",
        "checkpoint_sha256": "synthetic-checkpoint",
        "seed": 2025,
        "num_classes": 6,
        "num_tasks": 3,
        "diagnostic_tasks": 2,
        "validation_fraction": 0.25,
        "statistics_dtype": "float64",
        "solver_dtype": "float64",
        "ridge_lambda": 100.0,
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
            "maximum_gap_to_exact_fly_pp": 100.0,
            "maximum_state_fraction_of_exact_fly": 1.0,
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


def test_d0_runner_is_five_method_train_only_and_resumable(tmp_path):
    cache, config_path = _fixture(tmp_path)
    arguments = _args(tmp_path, cache, config_path)
    first = srq_fly_d0.run(arguments)
    second = srq_fly_d0.run(arguments)
    assert first["uses_test_set"] is False and first["held_out_test_authorized"] is False
    assert first["diagnostic_tasks"] == 2
    assert first["results"] == second["results"]
    assert {result["method"] for result in first["results"]} == {
        "exact_fly_10000", "exact_fly_4096", "raw_ridge",
        "direct_int8_gram", "sqrt_float16", "srq_int8",
    }
    assert all(result["exemplar_free"] for result in first["results"])
    assert first["gates"]["heldout_test_remained_hidden"] is True
    assert (tmp_path / "output" / "d0_results.json").is_file()


def test_d0_refuses_visible_test_cache(tmp_path):
    cache, config_path = _fixture(tmp_path)
    torch.save({"features": torch.zeros(1, 7)}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="held-out test.pt is visible"):
        srq_fly_d0.run(_args(tmp_path, cache, config_path))


def test_d0_config_is_strict_and_requires_seed_2025(tmp_path):
    config = _config()
    config["seed"] = 1993
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="seed 2025"):
        srq_fly_d0._read_config(path)
    config = _config()
    config["unexpected"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="keys/schema"):
        srq_fly_d0._read_config(path)
