import json
from pathlib import Path

import pytest
import torch

from tools import mars_soho_phase1b as runner


def _tiny_config():
    return {
        "schema_version": 1,
        "study_id": "mars-phase1b-tiny",
        "phase1_artifact": {
            "dataset": "Tiny",
            "sha256": "abc",
            "status": "phase1_failed",
            "locked_ridge_lambda": 1.0,
            "locked_covariance_rank": 2,
            "locked_shrinkage": 0.5,
        },
        "backbone": {
            "model_name": "tiny",
            "checkpoint_sha256": "abc",
            "feature_dim": 4,
            "preprocessing": "fixed",
        },
        "phase1b": {
            "expand_dim": 12,
            "olda_dim": 4,
            "split_seed": 3031,
            "outer_validation_fraction": 0.2,
            "inner_validation_fraction": 0.25,
            "ridge_lambda": 1.0,
            "reconstruction": {
                "covariance_rank": 2,
                "shrinkage": 0.5,
                "pseudo_per_class": 4,
                "pilot_per_class": 4,
                "minimum_per_class": 1,
                "risk_floor": 0.001,
            },
            "validation_replicates": [
                {"class_order_seed": 5101, "projection_seed": 31}
            ],
            "gates": {
                "max_oracle_gap_pp": 100.0,
                "min_uniform_gain_pp": -100.0,
                "min_shuffled_gain_pp": -100.0,
                "minimum_risk_spread": 0.0,
                "minimum_noncollapsed_fraction": 0.0,
                "minimum_distinct_allocation_fraction": 0.0,
            },
        },
        "datasets": {
            "tiny": {
                "dataset": "Tiny",
                "num_classes": 4,
                "num_tasks": 2,
                "train_samples": 48,
                "locked_soho": {
                    "density": 0.5,
                    "coding_level": 0.25,
                    "use_etf": True,
                },
            }
        },
    }


def _write_cache(path: Path):
    path.mkdir()
    labels = torch.arange(4).repeat_interleave(12)
    torch.manual_seed(71)
    features = torch.randn(48, 4)
    features += torch.nn.functional.one_hot(labels, 4).float()
    torch.save({"features": features, "labels": labels}, path / "train.pt")
    (path / "metadata.json").write_text(json.dumps({
        "dataset": "Tiny",
        "backbone_model": "tiny",
        "checkpoint_sha256": "abc",
        "preprocessing": "fixed",
    }))


def test_phase1b_rejects_changed_phase1_selection(tmp_path):
    config = _tiny_config()
    config["phase1b"]["ridge_lambda"] = 9.0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="locked Phase-1 selection"):
        runner._read_config(path)


def test_phase1b_fails_closed_when_test_is_visible(tmp_path):
    config = _tiny_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    cache = tmp_path / "cache"
    _write_cache(cache)
    torch.save({}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="refuses a visible test.pt"):
        runner.run(
            config_path=config_path,
            dataset_key="tiny",
            feature_cache_dir=cache,
            output_root=tmp_path / "output",
            device="cpu",
        )


def test_tiny_phase1b_runs_all_controls_and_resumes(tmp_path):
    config = _tiny_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    cache, output = tmp_path / "cache", tmp_path / "output"
    _write_cache(cache)
    first = runner.run(
        config_path=config_path,
        dataset_key="tiny",
        feature_cache_dir=cache,
        output_root=output,
        device="cpu",
    )
    assert first["uses_test_set"] is False
    assert first["status"] == "phase1b_pass"
    assert set(first["outer_validation"]) == set(runner.METHODS)
    assert first["outer_validation"]["exact_replay_oracle"][0][
        "exemplar_free"
    ] is False
    proposed = first["outer_validation"]["statistic_variance_aware"][0]
    assert proposed["exemplar_free"] is True
    assert proposed["state_audit"]["sample_level_bytes"] == 0
    assert proposed["task_diagnostics"][-1]["allocation_risk_name"] == (
        "statistic_variance"
    )
    assert not (cache / "test.pt").exists()
    second = runner.run(
        config_path=config_path,
        dataset_key="tiny",
        feature_cache_dir=cache,
        output_root=output,
        device="cpu",
    )
    assert second["outer_mean_aia"] == first["outer_mean_aia"]
