import argparse
import json
from pathlib import Path

import pytest
import torch

from tools import certifly_q1


def _config():
    return {
        "schema_version": 1,
        "study_id": "certifly-synthetic-q1",
        "dataset": "Synthetic",
        "model_name": "tiny_backbone",
        "checkpoint_sha256": "synthetic-checkpoint",
        "seed": 2025,
        "num_classes": 6,
        "num_tasks": 3,
        "validation_fraction": 0.25,
        "statistics_dtype": "float64",
        "solver_dtype": "float64",
        "representation": {
            "expand_dim": 18,
            "synaptic_degree": 3,
            "coding_level": 1 / 3,
            "encode_batch_size": 16,
            "evaluation_batch_size": 16,
        },
        "certifly_candidates": [
            {
                "name": "fixed_int8",
                "block_size": 8,
                "error_fraction": 0.99,
                "max_bits": 8,
            },
            {
                "name": "adaptive",
                "block_size": 8,
                "error_fraction": 0.10,
                "max_bits": 16,
            },
        ],
        "raw_ridge_lambdas": [0.1, 1.0],
        "fly_control": {
            "ridge_lower": 2,
            "ridge_upper": 4,
            "statistics_dtype": "float64",
        },
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
    torch.save(
        {"features": torch.cat(features), "labels": torch.cat(labels)},
        cache / "train.pt",
    )
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "Synthetic",
                "backbone_model": "tiny_backbone",
                "checkpoint_sha256": "synthetic-checkpoint",
                "feature_dim": 7,
                "finite": True,
                "test_features_materialized": False,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    return cache, config_path


def _args(tmp_path, cache, config_path):
    return argparse.Namespace(
        config=str(config_path),
        feature_cache_dir=str(cache),
        code_cache_dir=str(tmp_path / "wta"),
        output_dir=str(tmp_path / "output"),
        device="cpu",
        require_test_hidden=True,
    )


def test_q1_runner_is_train_only_and_resumable(tmp_path):
    cache, config_path = _fixture(tmp_path)
    arguments = _args(tmp_path, cache, config_path)
    first = certifly_q1.run(arguments)
    second = certifly_q1.run(arguments)

    assert first["uses_test_set"] is False
    assert first["held_out_test_authorized"] is False
    assert first["selected_certifly"] == second["selected_certifly"]
    assert first["gates"]["heldout_test_remained_hidden"] is True
    assert first["gates"]["numerical_stability"] is True
    assert first["selected_certifly"]["exemplar_free"] is True
    assert {row["candidate"]["name"] for row in first["certifly_candidates"]} == {
        "fixed_int8",
        "adaptive",
    }
    assert (tmp_path / "output" / "q1_results.json").is_file()


def test_q1_runner_refuses_visible_test_cache(tmp_path):
    cache, config_path = _fixture(tmp_path)
    torch.save({"features": torch.zeros(1, 7)}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="held-out test.pt is visible"):
        certifly_q1.run(_args(tmp_path, cache, config_path))


def test_q1_config_is_strict_and_requires_seed_2025(tmp_path):
    config = _config()
    config["seed"] = 1993
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="seed 2025"):
        certifly_q1._read_config(path)

    config = _config()
    config["unexpected"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config keys"):
        certifly_q1._read_config(path)
