import json
from pathlib import Path

import pytest
import torch

from tools import mars_soho_phase1c as runner


def _tiny_config():
    return {
        "schema_version": 1,
        "study_id": "mars-phase1c-tiny",
        "phase1b_artifact": {
            "dataset": "Tiny", "sha256": "abc", "status": "phase1b_failed"
        },
        "backbone": {
            "model_name": "tiny", "checkpoint_sha256": "abc",
            "feature_dim": 6, "preprocessing": "fixed",
        },
        "phase1c": {
            "expand_dim": 18, "olda_dim": 6,
            "outer_validation_fraction": 0.2,
            "inner_validation_fraction": 0.25,
            "ridge_lambda": 1.0, "pseudo_per_class": 8,
            "ambient_covariance_rank": 4, "ambient_shrinkage": 0.5,
            "tangent_rank_grid": [2, 3], "near_tie_tolerance": 0.002,
            "development_replicates": [
                {"split_seed": 6101, "projection_seed": 41}
            ],
            "gates": {
                "minimum_relative_stat_error_reduction": -100.0,
                "minimum_accuracy_gain_pp": -100.0,
                "maximum_accuracy_gap_to_empirical_oracle_pp": 100.0,
                "maximum_resultant_length_error": 1.0,
            },
        },
        "datasets": {
            "tiny": {
                "dataset": "Tiny", "num_classes": 4, "train_samples": 96,
                "locked_soho": {
                    "density": 0.5, "coding_level": 0.25, "use_etf": True,
                },
            }
        },
    }


def _write_cache(path: Path):
    path.mkdir()
    labels = torch.arange(4).repeat_interleave(24)
    torch.manual_seed(131)
    centers = torch.eye(6)[:4]
    features = torch.nn.functional.normalize(
        centers[labels] + 0.2 * torch.randn(96, 6), p=2, dim=1
    )
    torch.save({"features": features, "labels": labels}, path / "train.pt")
    (path / "metadata.json").write_text(json.dumps({
        "dataset": "Tiny", "backbone_model": "tiny",
        "checkpoint_sha256": "abc", "preprocessing": "fixed",
    }))


def test_phase1c_fails_closed_when_test_is_visible(tmp_path):
    config = _tiny_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    cache = tmp_path / "cache"
    _write_cache(cache)
    torch.save({}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="refuses a visible test.pt"):
        runner.run(
            config_path=config_path, dataset_key="tiny",
            feature_cache_dir=cache, output_root=tmp_path / "out", device="cpu",
        )


def test_tiny_phase1c_selects_rank_runs_controls_and_resumes(tmp_path):
    config = _tiny_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    cache, output = tmp_path / "cache", tmp_path / "output"
    _write_cache(cache)
    first = runner.run(
        config_path=config_path, dataset_key="tiny", feature_cache_dir=cache,
        output_root=output, device="cpu",
    )
    assert first["status"] == "phase1c_pass"
    assert first["uses_test_set"] is False
    assert first["selected_tangent_rank"] in {2, 3}
    assert set(first["outer_fidelity"]) == set(runner.METHODS)
    for method in runner.METHODS:
        result = first["outer_fidelity"][method][0]
        assert torch.isfinite(torch.tensor(result["combined_stat_error"]))
        if method == "empirical_replay_oracle":
            audit = result["state_audit"]
            assert audit["exemplar_free"] is False
            assert audit["historical_feature_rows"] > 0
            assert audit["sample_level_bytes"] > 0
            assert "feature_history" in audit["persistent_tensors"]
            assert "label_history" in audit["persistent_tensors"]
        else:
            assert result["state_audit"]["sample_level_bytes"] == 0
    assert not (cache / "test.pt").exists()
    second = runner.run(
        config_path=config_path, dataset_key="tiny", feature_cache_dir=cache,
        output_root=output, device="cpu",
    )
    assert second["outer_summary"] == first["outer_summary"]
