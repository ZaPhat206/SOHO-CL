import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from methods.cached_replay_baselines import CachedFlyCLFidelity
from methods.zi_soho import ZISOHOLearner
from tools import experiment_runner, zi_soho_pilot


def _fixture(tmp_path):
    feature_cache = tmp_path / "features"
    code_cache = tmp_path / "codes"
    output = tmp_path / "output"
    generator = torch.Generator().manual_seed(41)
    features = torch.randn(60, 6, generator=generator)
    labels = torch.arange(3).repeat_interleave(20)
    metadata = {
        "schema_version": 1,
        "dataset": "tiny-zi",
        "backbone_model": "synthetic",
        "checkpoint_sha256": "synthetic-checkpoint",
        "feature_dim": 6,
        "finite": True,
    }
    feature_cache.mkdir()
    (feature_cache / "metadata.json").write_text(json.dumps(metadata))
    torch.save({"features": features, "labels": labels}, feature_cache / "train.pt")
    torch.save({"features": features[:9], "labels": labels[:9]}, feature_cache / "test.pt")
    config = {
        "schema_version": 1,
        "study_id": "tiny-zi-train-only",
        "dataset": "tiny-zi",
        "model_name": "synthetic",
        "checkpoint_sha256": "synthetic-checkpoint",
        "seed": 7,
        "num_classes": 3,
        "num_tasks": 3,
        "validation_fraction": 0.2,
        "representation": {
            "expand_dim": 12,
            "synaptic_degree": 3,
            "coding_level": 0.25,
            "encode_batch_size": 11,
            "score_chunk_size": 2,
        },
        "raw_ridge_lambda": 0.4,
        "fly_ridge_lower": -1,
        "fly_ridge_upper": 2,
        "support_alpha": 0.5,
        "variance_kappas": [1.0, 3.0],
        "variance_epsilon": 1e-4,
        "statistics_dtype": "float32",
        "gate": {
            "minimum_gain_over_raw_pp": 0.2,
            "maximum_gap_to_fly_pp": 0.5,
            "minimum_gain_over_component_pp": 0.1,
            "maximum_state_fraction_of_fly": 0.15,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    args = Namespace(
        config=str(config_path), feature_cache_dir=str(feature_cache),
        code_cache_dir=str(code_cache), output_dir=str(output), device="cpu",
        require_test_hidden=True, resume=False,
    )
    return args, feature_cache, code_cache, output, features


def test_runner_requires_physically_hidden_test_and_never_opens_it(tmp_path):
    args, feature_cache, code_cache, output, _ = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="held-out file is visible"):
        zi_soho_pilot.run(args)
    assert not code_cache.exists()
    (feature_cache / "test.pt").rename(feature_cache / "test.locked.pt")
    payload = zi_soho_pilot.run(args)
    assert payload["uses_test_set"] is False
    assert payload["held_out_test_authorized"] is False
    assert payload["run_provenance"]["heldout_test_path_visible"] is False
    assert (feature_cache / "test.locked.pt").is_file()
    assert (output / "selection.json").is_file()
    assert (output / "gate_results.json").is_file()


def test_pilot_candidate_grid_state_audit_and_resume(tmp_path, capsys):
    args, feature_cache, _, output, _ = _fixture(tmp_path)
    (feature_cache / "test.pt").rename(feature_cache / "test.locked.pt")
    payload = zi_soho_pilot.run(args)
    assert len(payload["candidates"]) == 8
    assert {entry["method"] for entry in payload["candidates"]} == {
        "sft_raw_ridge", "cached_flycl_fidelity", "wta_ncm", "support_only",
        "active_gaussian", "hurdle",
    }
    assert all(entry["uses_test_set"] is False for entry in payload["candidates"])
    assert all(entry["exemplar_free"] is True for entry in payload["candidates"])
    hurdle = payload["best_by_method"]["hurdle"]
    shapes = hurdle["persistent_tensor_manifest"]
    assert shapes["active_counts"]["shape"] == [12, 3]
    assert all(60 not in field["shape"] for field in shapes.values())
    assert payload["code_cache"]["role"] == "experiment_cache_not_learner_state"
    assert payload["code_cache"]["contains_sample_level_codes"] is True
    assert len(payload["run_provenance"]["run_identity_sha256"]) == 64
    assert payload["best_by_method"]["cached_flycl_fidelity"]["diagnostics"]["ridge_policy"] == "original_current_task_gcv"
    assert payload["gate"]["decision"] in {
        "REVIEW_FOR_HELDOUT_AUTHORIZATION", "STOP_TRAIN_ONLY_GATE_FAILED"
    }
    args.resume = True
    zi_soho_pilot.run(args)
    assert "SKIP 8/8" in capsys.readouterr().out
    assert len(list((output / "candidates").glob("*.json"))) == 8


def test_cached_sparse_codes_and_fly_use_identical_projection(tmp_path):
    args, feature_cache, code_cache, _, features = _fixture(tmp_path)
    (feature_cache / "test.pt").rename(feature_cache / "test.locked.pt")
    config = zi_soho_pilot._read_config(Path(args.config))
    train, _, _ = experiment_runner.validate_cache(
        feature_cache,
        Namespace(dataset="tiny-zi", model_name="synthetic"),
        load_test=False,
    )
    indices, values, _ = zi_soho_pilot._prepare_code_cache(
        train=train,
        train_sha256=zi_soho_pilot._sha256_file(feature_cache / "train.pt"),
        cache_dir=code_cache,
        config=config,
        device="cpu",
    )
    zi = zi_soho_pilot._new_zi(config, 6, "hurdle", 1.0, "cpu")
    direct_indices, direct_values = zi.encode_sparse(features)
    torch.testing.assert_close(indices.long(), direct_indices)
    torch.testing.assert_close(values, direct_values)
    fly = CachedFlyCLFidelity(
        feature_dim=6, expand_dim=12, synaptic_degree=3, coding_level=.25,
        num_classes=3, ridge_lower=-1, ridge_upper=2, seed=7, device="cpu",
    )
    torch.testing.assert_close(
        zi.projection.to_dense(), fly.flyhash.projection_matrix.to_dense(),
        atol=0, rtol=0,
    )
    dense = torch.zeros_like(fly._encode(features))
    dense.scatter_(1, indices.long(), values)
    torch.testing.assert_close(dense, fly._encode(features), atol=0, rtol=0)


def test_stale_code_cache_is_rejected_instead_of_silently_overwritten(tmp_path):
    args, feature_cache, code_cache, _, _ = _fixture(tmp_path)
    (feature_cache / "test.pt").rename(feature_cache / "test.locked.pt")
    config = zi_soho_pilot._read_config(Path(args.config))
    train, _, _ = experiment_runner.validate_cache(
        feature_cache, Namespace(dataset="tiny-zi", model_name="synthetic"),
        load_test=False,
    )
    zi_soho_pilot._prepare_code_cache(
        train=train,
        train_sha256=zi_soho_pilot._sha256_file(feature_cache / "train.pt"),
        cache_dir=code_cache, config=config, device="cpu",
    )
    metadata_path = code_cache / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["identity_sha256"] = "wrong"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="stale WTA code cache"):
        zi_soho_pilot._prepare_code_cache(
            train=train,
            train_sha256=zi_soho_pilot._sha256_file(feature_cache / "train.pt"),
            cache_dir=code_cache, config=config, device="cpu",
        )
