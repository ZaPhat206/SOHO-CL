import argparse
import json
from pathlib import Path

import pytest
import torch

from tools import tail_fly_phasea


def _config(*, a3=False):
    config = {
        "schema_version": 1,
        "study_id": "tail-fly-synthetic-train-only",
        "dataset": "Synthetic",
        "model_name": "tiny_backbone",
        "checkpoint_sha256": "synthetic-checkpoint",
        "seed": 2025,
        "num_classes": 6,
        "num_tasks": 3,
        "validation_fraction": 0.25,
        "statistics_dtype": "float64",
        "representation": {
            "expand_dim": 18,
            "synaptic_degree": 3,
            "coding_level": 1 / 3,
            "encode_batch_size": 16,
            "evaluation_batch_size": 16,
            "svd_update_batch_size": 8,
        },
        "search": {
            "ranks": [3, 6],
            "ridge_lambdas": [0.1, 1.0],
            "raw_ridge_lambdas": [0.1, 1.0],
        },
        "fly_control": {
            "ridge_lower": -1,
            "ridge_upper": 2,
            "statistics_dtype": "float64",
        },
        "gates": {
            "maximum_solver_relative_residual": 1e-5,
            "minimum_tail_gain_over_plain_tsvd_pp": -100.0,
            "maximum_gap_to_exact_fly_pp": 100.0,
            "minimum_gain_over_raw_ridge_pp": -100.0,
            "maximum_state_fraction_of_exact_fly": 10.0,
        },
    }
    if a3:
        config["study_id"] = "tail-fly-synthetic-train-only-a3"
        config["statistics_dtype"] = "float32"
        config["numerics"] = {"solver_dtype": "float64"}
        config["gates"]["minimum_gain_over_best_plain_tsvd_pp"] = -100.0
    return config


def _write_fixture(tmp_path: Path):
    cache = tmp_path / "features"
    cache.mkdir()
    generator = torch.Generator().manual_seed(41)
    features, labels = [], []
    for class_id in range(6):
        center = torch.randn(7, generator=generator) + class_id / 4
        features.append(center + 0.2 * torch.randn(8, 7, generator=generator))
        labels.append(torch.full((8,), class_id, dtype=torch.long))
    train = {"features": torch.cat(features), "labels": torch.cat(labels)}
    torch.save(train, cache / "train.pt")
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


def test_train_only_runner_completes_and_resumes_without_test_access(tmp_path):
    cache, config_path = _write_fixture(tmp_path)
    args = _args(tmp_path, cache, config_path)
    first = tail_fly_phasea.run(args)
    second = tail_fly_phasea.run(args)
    assert first["uses_test_set"] is False
    assert first["held_out_test_authorized"] is False
    assert first["selected_tail_config"] == second["selected_tail_config"]
    assert first["gates"]["numerical_stability"] is True
    assert first["gates"]["heldout_test_remained_hidden"] is True
    assert {row["method"] for row in first["candidates"]} == {
        "tail_fly",
        "plain_tsvd_fly",
        "diagonal_only_fly",
        "raw_ridge",
        "matched_exact_fly",
    }
    assert (tmp_path / "output" / "phasea_results.json").is_file()
    code_metadata = json.loads((tmp_path / "wta" / "metadata.json").read_text())
    assert code_metadata["role"] == "experiment_cache_not_learner_state"
    assert code_metadata["contains_sample_level_codes"] is True


def test_a3_runner_records_implementation_identity_and_per_method_residuals(tmp_path):
    cache, config_path = _write_fixture(tmp_path)
    config_path.write_text(json.dumps(_config(a3=True)), encoding="utf-8")
    result = tail_fly_phasea.run(_args(tmp_path, cache, config_path))
    assert result["selected_plain_tsvd"]["method"] == "plain_tsvd_fly"
    assert "beats_independently_selected_plain_tsvd" in result["gates"]
    context = result["provenance"]
    for key in (
        "runner_git_commit",
        "runner_source_sha256",
        "solver_source_sha256",
        "learner_source_sha256",
        "streaming_svd_source_sha256",
    ):
        assert context[key]
    rank_unit = json.loads(
        (tmp_path / "output" / "units" / "rank_3.json").read_text()
    )
    assert set(rank_unit["task_diagnostics"][0]["solver_relative_residuals"]) == {
        "tail_fly",
        "plain_tsvd_fly",
        "diagonal_only_fly",
    }
    for candidate in rank_unit["candidates"]:
        assert candidate["maximum_solver_relative_residual"] < 1e-5


def test_runner_refuses_visible_heldout_file(tmp_path):
    cache, config_path = _write_fixture(tmp_path)
    torch.save({"features": torch.zeros(1, 7), "labels": torch.zeros(1)}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="held-out file is visible"):
        tail_fly_phasea.run(_args(tmp_path, cache, config_path))


def test_unit_resume_refuses_a_different_implementation_context(tmp_path):
    path = tmp_path / "rank_64.json"
    tail_fly_phasea._save_unit(path, "implementation-a", {"rank": 64})
    with pytest.raises(RuntimeError, match="stale unit artifact"):
        tail_fly_phasea._load_unit(path, "implementation-b")


def test_config_is_strict_and_requires_repository_seed(tmp_path):
    config = _config()
    config["seed"] = 1993
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="seed 2025"):
        tail_fly_phasea._read_config(path)
    config = _config()
    config["unexpected"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config keys"):
        tail_fly_phasea._read_config(path)
