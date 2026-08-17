import json
from argparse import Namespace

import pytest
import torch

from tools import twa_fly_pilot


def fixture(tmp_path):
    feature_cache = tmp_path / "features"
    code_cache = tmp_path / "codes"
    output = tmp_path / "output"
    generator = torch.Generator().manual_seed(71)
    features = torch.randn(72, 6, generator=generator)
    labels = torch.arange(4).repeat_interleave(18)
    metadata = {
        "schema_version": 1,
        "dataset": "tiny-twa",
        "backbone_model": "synthetic",
        "checkpoint_sha256": "synthetic-checkpoint",
        "feature_dim": 6,
        "finite": True,
    }
    feature_cache.mkdir()
    (feature_cache / "metadata.json").write_text(json.dumps(metadata))
    torch.save({"features": features, "labels": labels}, feature_cache / "train.pt")
    torch.save({"features": features[:8], "labels": labels[:8]}, feature_cache / "test.pt")
    config = {
        "schema_version": 1,
        "study_id": "tiny-twa-train-only",
        "dataset": "tiny-twa",
        "model_name": "synthetic",
        "checkpoint_sha256": "synthetic-checkpoint",
        "seed": 7,
        "num_classes": 4,
        "num_tasks": 2,
        "validation_fraction": 0.2,
        "representation": {
            "expand_dim": 12,
            "synaptic_degree": 3,
            "coding_level": 0.25,
            "encode_batch_size": 11,
        },
        "raw_ridge_lambda": 0.4,
        "fly_ridge_lower": -1,
        "fly_ridge_upper": 2,
        "rho_candidates": [0.05, 0.2],
        "solver_tolerance": 1e-7,
        "solver_max_iterations": 300,
        "statistics_dtype": "float64",
        "gate": {
            "minimum_gain_over_fly_pp": -100.0,
            "minimum_gain_over_one_way_pp": -100.0,
            "minimum_gain_over_shuffled_pp": -100.0,
            "maximum_state_fraction_of_fly": 10.0,
            "maximum_solver_relative_residual": 1e-7,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    args = Namespace(
        config=str(config_path), feature_cache_dir=str(feature_cache),
        code_cache_dir=str(code_cache), output_dir=str(output),
        device="cpu", require_test_hidden=True,
    )
    return args, feature_cache, code_cache, output


def test_pilot_requires_physically_hidden_test(tmp_path):
    args, feature_cache, code_cache, _ = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="held-out file is visible"):
        twa_fly_pilot.run(args)
    assert not code_cache.exists()
    (feature_cache / "test.pt").rename(feature_cache / "test.locked.pt")
    payload = twa_fly_pilot.run(args)
    assert payload["uses_test_set"] is False
    assert payload["held_out_test_authorized"] is False
    assert payload["run_provenance"]["heldout_test_path_visible"] is False
    assert (feature_cache / "test.locked.pt").is_file()


def test_pilot_grid_statistics_state_and_metric_definition(tmp_path):
    args, feature_cache, _, output = fixture(tmp_path)
    (feature_cache / "test.pt").rename(feature_cache / "test.locked.pt")
    payload = twa_fly_pilot.run(args)
    assert len(payload["candidates"]) == 8
    assert {row["method"] for row in payload["candidates"]} == {
        "matched_fly", "raw_ridge", "twa_one_way", "twa_symmetric", "twa_shuffled_cross"
    }
    assert all(row["uses_test_set"] is False and row["exemplar_free"] is True for row in payload["candidates"])
    for row in payload["candidates"]:
        assert row["validation_average_accuracy"] == pytest.approx(
            sum(row["stage_average_accuracy"]) / len(row["stage_average_accuracy"])
        )
    selected = payload["gate"]["selected_symmetric"]
    assert max(selected["solver_relative_residual"]) <= 1e-7
    assert payload["code_cache"]["role"] == "experiment_cache_not_learner_state"
    assert payload["code_cache"]["contains_sample_level_codes"] is True
    assert (output / "selection.json").is_file()
    assert (output / "gate_results.json").is_file()


def test_code_cache_restores_and_matches_dense_projection(tmp_path):
    args, feature_cache, code_cache, _ = fixture(tmp_path)
    config = twa_fly_pilot._read_config(tmp_path / "config.json")
    train = torch.load(feature_cache / "train.pt", weights_only=True)
    train_sha = twa_fly_pilot._sha256_file(feature_cache / "train.pt")
    indices, values, metadata, projection = twa_fly_pilot._prepare_code_cache(
        train=train, train_sha256=train_sha, cache_dir=code_cache, config=config, device="cpu"
    )
    learner = twa_fly_pilot._new_learner(config, 6, "twa_symmetric", 0.0, "cpu")
    dense = twa_fly_pilot._dense_codes(indices, values, 12, "cpu", torch.float64)
    torch.testing.assert_close(dense, learner.encode_fly(train["features"]))
    restored_indices, restored_values, restored_metadata, restored_projection = twa_fly_pilot._prepare_code_cache(
        train=train, train_sha256=train_sha, cache_dir=code_cache, config=config, device="cpu"
    )
    torch.testing.assert_close(restored_indices, indices)
    torch.testing.assert_close(restored_values, values)
    torch.testing.assert_close(restored_projection.to_dense(), projection.to_dense())
    assert restored_metadata["identity_sha256"] == metadata["identity_sha256"]
    assert (code_cache / "projection.pt").is_file()
