import json
from pathlib import Path

import pytest
import torch

from tools import mars_soho_phase1 as runner


def _tiny_config():
    return {
        "schema_version": 1,
        "study_id": "mars-tiny",
        "backbone": {
            "model_name": "tiny", "checkpoint_sha256": "abc",
            "feature_dim": 4, "preprocessing": "fixed",
        },
        "phase1": {
            "expand_dim": 12, "olda_dim": 4, "split_seed": 2025,
            "outer_validation_fraction": 0.2,
            "inner_validation_fraction": 0.25,
            "ridge_grid": [0.1, 1.0], "near_tie_tolerance_pp": 0.05,
            "reconstruction_grid": {
                "covariance_rank": [2], "shrinkage": [0.5],
                "pseudo_per_class": [4], "pilot_per_class": 4,
                "minimum_per_class": 1, "risk_floor": 0.001,
            },
            "development_replicates": [
                {"class_order_seed": 2025, "projection_seed": 11}
            ],
            "gates": {
                "max_oracle_gap_pp": 100.0,
                "min_shared_gain_pp": -100.0,
                "min_shuffled_gain_pp": -100.0,
            },
        },
        "datasets": {
            "tiny": {
                "dataset": "Tiny", "num_classes": 4, "num_tasks": 2,
                "train_samples": 48,
                "locked_soho": {
                    "density": 0.5, "coding_level": 0.25, "use_etf": True,
                },
            }
        },
    }


def _write_cache(path: Path, config: dict):
    path.mkdir()
    labels = torch.arange(4).repeat_interleave(12)
    torch.manual_seed(23)
    features = torch.randn(48, 4)
    features += torch.nn.functional.one_hot(labels, 4).float()
    torch.save({"features": features, "labels": labels}, path / "train.pt")
    (path / "metadata.json").write_text(json.dumps({
        "dataset": "Tiny", "backbone_model": "tiny",
        "checkpoint_sha256": "abc", "preprocessing": "fixed",
    }))


def test_phase1_cache_fails_closed_when_test_is_visible(tmp_path):
    config = _tiny_config()
    cache = tmp_path / "cache"
    _write_cache(cache, config)
    torch.save({}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="refuses a visible test.pt"):
        runner._validate_train_cache(cache, config, "tiny")


def test_candidate_near_tie_prefers_smaller_reconstruction():
    results = [
        {"mean_inner_aia": 80.04, "candidate": {
            "pseudo_per_class": 64, "covariance_rank": 32, "shrinkage": 0.1,
        }},
        {"mean_inner_aia": 80.00, "candidate": {
            "pseudo_per_class": 32, "covariance_rank": 32, "shrinkage": 0.5,
        }},
    ]
    selected = runner._select_candidate(results, tolerance_pp=0.05)
    assert selected is results[1]


def test_tiny_phase1_runs_nested_train_only_and_resumes(tmp_path):
    config = _tiny_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    cache, output = tmp_path / "cache", tmp_path / "output"
    _write_cache(cache, config)
    first = runner.run(
        config_path=config_path, dataset_key="tiny", feature_cache_dir=cache,
        output_root=output, device="cpu",
    )
    assert first["uses_test_set"] is False
    assert first["status"] == "phase1_pass"
    assert set(first["outer_validation"]) == set(runner.METHODS)
    assert first["outer_validation"]["exact_replay_oracle"][0]["exemplar_free"] is False
    assert first["outer_validation"]["support_aware"][0]["exemplar_free"] is True
    assert first["outer_validation"]["exact_replay_oracle"][0]["state_audit"]["sample_level_bytes"] > 0
    assert first["outer_validation"]["support_aware"][0]["state_audit"]["sample_level_bytes"] == 0
    assert not (cache / "test.pt").exists()
    second = runner.run(
        config_path=config_path, dataset_key="tiny", feature_cache_dir=cache,
        output_root=output, device="cpu",
    )
    assert second["outer_mean_aia"] == first["outer_mean_aia"]
