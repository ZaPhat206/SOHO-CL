import json
from pathlib import Path

import pytest
import torch

from tools import wbt_soho_phase1d as runner


def _tiny_config():
    return {
        "schema_version": 1,
        "study_id": "wbt-phase1d-tiny",
        "phase1c_artifact": {
            "dataset": "Tiny",
            "sha256": "abc",
            "status": "phase1c_failed",
            "selected_tangent_rank": 2,
        },
        "backbone": {
            "model_name": "tiny",
            "checkpoint_sha256": "abc",
            "feature_dim": 6,
            "preprocessing": "fixed",
        },
        "phase1d": {
            "expand_dim": 18,
            "olda_dim": 6,
            "split_seed": 2025,
            "outer_validation_fraction": 0.2,
            "inner_validation_fraction": 0.25,
            "ridge_lambda": 1.0,
            "tangent_rank": 2,
            "pseudo_per_class": 8,
            "near_tie_tolerance_pp": 0.05,
            "boundary_grid": {
                "boundary_fraction": [0.5],
                "boundary_strength": [0.5],
            },
            "development_replicates": [
                {"class_order_seed": 6101, "projection_seed": 41}
            ],
            "gates": {
                "minimum_oracle_gap_closed_fraction": -100.0,
                "minimum_relative_stat_error_reduction": -100.0,
                "minimum_shuffled_gain_pp": -100.0,
                "maximum_oracle_gap_pp": 100.0,
                "minimum_old_dominance_fraction": 0.0,
                "maximum_solver_relative_residual": 1.0,
            },
        },
        "datasets": {
            "tiny": {
                "dataset": "Tiny",
                "num_classes": 4,
                "num_tasks": 2,
                "train_samples": 96,
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
    labels = torch.arange(4).repeat_interleave(24)
    generator = torch.Generator().manual_seed(719)
    centers = torch.eye(6)[:4]
    features = torch.nn.functional.normalize(
        centers[labels] + 0.2 * torch.randn(96, 6, generator=generator),
        p=2,
        dim=1,
    )
    torch.save({"features": features, "labels": labels}, path / "train.pt")
    (path / "metadata.json").write_text(json.dumps({
        "dataset": "Tiny",
        "backbone_model": "tiny",
        "checkpoint_sha256": "abc",
        "preprocessing": "fixed",
    }))


def test_phase1d_refuses_visible_test_cache(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_tiny_config()))
    cache = tmp_path / "cache"
    _write_cache(cache)
    torch.save({}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="refuses a visible test.pt"):
        runner.run(
            config_path=config_path,
            dataset_key="tiny",
            feature_cache_dir=cache,
            output_root=tmp_path / "out",
            device="cpu",
        )


def test_tiny_phase1d_runs_controls_audits_state_and_resumes(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_tiny_config()))
    cache, output = tmp_path / "cache", tmp_path / "output"
    _write_cache(cache)
    first = runner.run(
        config_path=config_path,
        dataset_key="tiny",
        feature_cache_dir=cache,
        output_root=output,
        device="cpu",
    )
    assert first["status"] in {"phase1d_pass", "phase1d_failed"}
    assert first["uses_test_set"] is False
    assert set(first["outer_validation"]) == set(runner.METHODS)
    assert first["selected_boundary"] == {
        "boundary_fraction": 0.5,
        "boundary_strength": 0.5,
    }
    oracle = first["outer_validation"]["exact_replay_oracle"][0]
    assert oracle["state_audit"]["sample_level_bytes"] > 0
    for method in runner.METHODS[1:]:
        result = first["outer_validation"][method][0]
        assert result["exemplar_free"] is True
        assert result["state_audit"]["sample_level_bytes"] == 0
        assert torch.isfinite(torch.tensor(result["mean_combined_stat_error"]))
    assert not (cache / "test.pt").exists()
    second = runner.run(
        config_path=config_path,
        dataset_key="tiny",
        feature_cache_dir=cache,
        output_root=output,
        device="cpu",
    )
    assert second["outer_summary"] == first["outer_summary"]
