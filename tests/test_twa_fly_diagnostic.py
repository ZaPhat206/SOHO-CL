import json
from argparse import Namespace

import pytest
import torch

from tools import twa_fly_diagnostic, twa_fly_pilot


def fixture(tmp_path):
    feature_cache = tmp_path / "features"
    code_cache = tmp_path / "codes"
    output = tmp_path / "output"
    generator = torch.Generator().manual_seed(89)
    features = torch.randn(72, 6, generator=generator)
    labels = torch.arange(4).repeat_interleave(18)
    feature_cache.mkdir()
    (feature_cache / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "dataset": "tiny-d0",
        "backbone_model": "synthetic",
        "checkpoint_sha256": "synthetic-checkpoint",
        "feature_dim": 6,
        "finite": True,
    }))
    torch.save({"features": features, "labels": labels}, feature_cache / "train.pt")
    torch.save({"features": features[:8], "labels": labels[:8]}, feature_cache / "test.pt")
    config = {
        "schema_version": 1,
        "study_id": "tiny-twa-d0",
        "dataset": "tiny-d0",
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
        "raw_ridge_lambda": 0.01,
        "fly_ridge_lower": -1,
        "fly_ridge_upper": 2,
        "fusion_alphas": [0.0, 0.1, 1.0],
        "solver_tolerance": 1e-8,
        "solver_max_iterations": 100,
        "statistics_dtype": "float64",
        "gate": {
            "required_raw_ridge_lambda": 0.01,
            "minimum_oracle_headroom_pp": -100.0,
            "minimum_fusion_gain_pp": -100.0,
            "maximum_solver_relative_residual": 1e-8,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return Namespace(
        config=str(config_path),
        feature_cache_dir=str(feature_cache),
        code_cache_dir=str(code_cache),
        output_dir=str(output),
        device="cpu",
        require_test_hidden=True,
    ), feature_cache, code_cache, output


def test_complementarity_metrics_counts_errors_and_alpha_zero_is_fly():
    labels = torch.tensor([0, 1, 2, 1])
    raw_logits = torch.tensor([
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 4.0],
    ])
    fly_logits = torch.tensor([
        [2.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 2.0, 0.0],
    ])
    result = twa_fly_diagnostic.complementarity_metrics(
        raw_logits, fly_logits, labels,
        raw_scale=1.0, fly_scale=1.0, fusion_alphas=[0.0, 1.0],
    )
    assert result["both_correct"] == 1
    assert result["raw_only_correct"] == 1
    assert result["fly_only_correct"] == 2
    assert result["both_wrong"] == 0
    assert result["fly_accuracy"] == 75.0
    assert result["raw_accuracy"] == 50.0
    assert result["oracle_union_accuracy"] == 100.0
    assert result["oracle_headroom_over_fly_pp"] == 25.0
    assert result["fusion_accuracy"]["0.0"] == result["fly_accuracy"]


def test_d0_requires_hidden_test_and_writes_train_only_evidence(tmp_path):
    args, feature_cache, _, output = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="held-out file is visible"):
        twa_fly_diagnostic.run(args)
    (feature_cache / "test.pt").rename(feature_cache / "test.locked.pt")
    payload = twa_fly_diagnostic.run(args)
    assert payload["uses_test_set"] is False
    assert payload["held_out_test_authorized"] is False
    assert len(payload["stage_diagnostics"]) == 2
    assert payload["code_cache"]["schema_version"] == 2
    assert payload["code_cache"]["projection"]["probe"]["verified"] is True
    assert payload["run_provenance"]["heldout_test_path_visible"] is False
    assert payload["gate"]["gates"]["raw_ridge_matches_locked_protocol"] is True
    assert (output / "diagnostics.json").is_file()
    assert (output / "gate_results.json").is_file()
    assert (feature_cache / "test.locked.pt").is_file()


def test_legacy_code_cache_is_upgraded_only_after_projection_probe(tmp_path):
    args, feature_cache, code_cache, _ = fixture(tmp_path)
    config = twa_fly_diagnostic._read_config(tmp_path / "config.json")
    train = torch.load(feature_cache / "train.pt", weights_only=True)
    train_sha = twa_fly_pilot._sha256_file(feature_cache / "train.pt")
    _, _, metadata, _ = twa_fly_pilot._prepare_code_cache(
        train=train, train_sha256=train_sha, cache_dir=code_cache,
        config=config, device="cpu",
    )
    legacy = dict(metadata)
    legacy["schema_version"] = 1
    legacy.pop("projection")
    (code_cache / "metadata.json").write_text(json.dumps(legacy))
    (code_cache / "projection.pt").unlink()
    _, _, upgraded, _ = twa_fly_pilot._prepare_code_cache(
        train=train, train_sha256=train_sha, cache_dir=code_cache,
        config=config, device="cpu",
    )
    assert upgraded["schema_version"] == 2
    assert upgraded["projection"]["probe"]["verified"] is True
    on_disk = json.loads((code_cache / "metadata.json").read_text())
    assert on_disk["projection"]["sha256"] == upgraded["projection"]["sha256"]
    assert (code_cache / "projection.pt").is_file()


def test_projection_hash_mismatch_fails_closed(tmp_path):
    args, feature_cache, code_cache, _ = fixture(tmp_path)
    config = twa_fly_diagnostic._read_config(tmp_path / "config.json")
    train = torch.load(feature_cache / "train.pt", weights_only=True)
    train_sha = twa_fly_pilot._sha256_file(feature_cache / "train.pt")
    twa_fly_pilot._prepare_code_cache(
        train=train, train_sha256=train_sha, cache_dir=code_cache,
        config=config, device="cpu",
    )
    metadata_path = code_cache / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["projection"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="projection SHA-256 mismatch"):
        twa_fly_pilot._prepare_code_cache(
            train=train, train_sha256=train_sha, cache_dir=code_cache,
            config=config, device="cpu",
        )
