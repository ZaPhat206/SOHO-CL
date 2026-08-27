import json
from pathlib import Path

import pytest
import torch

from tools import mt_soho_phase1 as runner


def _config():
    return {
        "schema_version": 1,
        "study_id": "mt-soho-tiny",
        "backbone": {
            "model_name": "tiny",
            "checkpoint_sha256": "abc",
            "feature_dim": 6,
            "preprocessing": "fixed",
        },
        "dataset": {
            "name": "Tiny",
            "num_classes": 4,
            "num_tasks": 2,
            "train_samples": 80,
        },
        "phase1": {
            "expand_dim": 18,
            "split_seed": 2025,
            "outer_validation_fraction": 0.2,
            "inner_validation_fraction": 0.25,
            "near_tie_tolerance_pp": 0.05,
            "geometry_epsilon": 1e-6,
            "anchor": {"synaptic_degree": 3, "coding_level": 1 / 3},
            "anchor_ridge_grid": [0.5],
            "target_grid": {
                "rank": [2],
                "shrinkage": [0.2],
                "adapted_ridge": [0.25],
                "adaptation_weight": [0.5],
            },
            "development_replicates": [
                {"class_order_seed": 2025, "projection_seed": 31}
            ],
            "gates": {
                "max_solver_relative_residual": 1e-3,
                "min_fixed_gain_pp": -100.0,
                "min_shuffled_gain_pp": -100.0,
                "min_whitening_gain_pp": -100.0,
            },
        },
    }


def _cache(path: Path):
    path.mkdir()
    generator = torch.Generator().manual_seed(9)
    labels = torch.arange(4).repeat_interleave(20)
    features = torch.randn((80, 6), generator=generator)
    features[:, :4] += 1.5 * torch.nn.functional.one_hot(labels, 4).float()
    torch.save({"features": features, "labels": labels}, path / "train.pt")
    (path / "metadata.json").write_text(json.dumps({
        "dataset": "Tiny",
        "backbone_model": "tiny",
        "checkpoint_sha256": "abc",
        "preprocessing": "fixed",
    }))


def test_cache_validation_fails_closed_when_test_is_visible(tmp_path):
    cache = tmp_path / "cache"
    _cache(cache)
    torch.save({}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="refuses a visible test.pt"):
        runner._validate_train_cache(cache, _config())


def test_candidate_grid_is_predeclared_cartesian_product():
    candidates = runner._candidate_grid(_config(), anchor_ridge=0.5)
    assert candidates == [{
        "anchor_ridge": 0.5,
        "projection_ridge": 0.5,
        "adapted_ridge": 0.25,
        "target_rank": 2,
        "shrinkage": 0.2,
        "adaptation_weight": 0.5,
    }]


def test_tiny_train_only_study_runs_and_resumes(tmp_path):
    config = _config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    cache, output = tmp_path / "cache", tmp_path / "output"
    _cache(cache)
    first = runner.run(
        config_path=config_path,
        feature_cache_dir=cache,
        output_dir=output,
        device="cpu",
    )
    assert first["status"] == "phase1a_pass"
    assert first["uses_test_set"] is False
    assert set(first["outer_validation"]) == set(runner.OUTER_METHODS)
    assert all(
        item["exemplar_free"]
        for values in first["outer_validation"].values()
        for item in values
    )
    assert not (cache / "test.pt").exists()
    second = runner.run(
        config_path=config_path,
        feature_cache_dir=cache,
        output_dir=output,
        device="cpu",
    )
    assert second["outer_mean_aia"] == first["outer_mean_aia"]
