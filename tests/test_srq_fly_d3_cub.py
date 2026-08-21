"""Synthetic correctness tests for the locked CUB SRQ-FLY D3 runner."""

import argparse
import json
from pathlib import Path

import pytest
import torch

from methods.srq_fly import projected_srq_state_bytes
from tools import srq_fly_d3_cub as d3


def _write_fixture(tmp_path: Path):
    feature_dim, classes, tasks = 7, 200, 20
    generator = torch.Generator().manual_seed(44)
    features, labels = [], []
    for class_id in range(classes):
        center = torch.randn(feature_dim, generator=generator) + class_id / 50
        features.append(center + 0.1 * torch.randn(6, feature_dim, generator=generator))
        labels.append(torch.full((6,), class_id, dtype=torch.long))
    train = {"features": torch.cat(features), "labels": torch.cat(labels)}
    cache = tmp_path / "features"
    cache.mkdir()
    torch.save(train, cache / "train.pt")
    (cache / "metadata.json").write_text(json.dumps({
        "schema_version": 1, "dataset": "CUB-200-2011",
        "dataset_version": "processed-imagefolder",
        "backbone_model": "tiny_backbone", "checkpoint_sha256": "synthetic",
        "preprocessing": "vit", "feature_dim": feature_dim, "finite": True,
        "test_features_materialized": False,
        "split_sizes": {"train": len(train["labels"]), "test": 400},
    }), encoding="utf-8")
    identity = {
        "dataset": "CUB-200-2011", "dataset_version": "processed-imagefolder",
        "dataset_identity_sha256": "synthetic-dataset",
        "class_mapping_sha256": "synthetic-mapping",
        "train_content_manifest_sha256": "synthetic-train",
        "test_content_manifest_sha256": "synthetic-test",
        "train_samples": len(train["labels"]), "test_samples": 400,
    }
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "dataset": "CUB-200-2011", "dataset_identity_sha256": "synthetic-dataset",
        "class_mapping_sha256": "synthetic-mapping",
        "cross_split_duplicate_content_count": 0,
        "train": {"image_count": len(train["labels"]), "content_manifest_sha256": "synthetic-train"},
        "test": {"image_count": 400, "content_manifest_sha256": "synthetic-test"},
    }), encoding="utf-8")
    large = {
        "expand_dim": 12, "synaptic_degree": 3, "coding_level": 0.25,
        "encode_batch_size": 128, "evaluation_batch_size": 128,
    }
    matched = {
        "expand_dim": 6, "synaptic_degree": 3, "coding_level": 0.25,
        "encode_batch_size": 128, "evaluation_batch_size": 128,
    }
    storage = {"block_size": 4, "group_size": 2}
    large_nnz = large["expand_dim"] * large["synaptic_degree"]
    matched_nnz = matched["expand_dim"] * matched["synaptic_degree"]
    exact_large = d3._projection_state_bytes(
        feature_dim=feature_dim, expand_dim=large["expand_dim"], nonzeros=large_nnz,
        num_classes=classes, gram_or_factor_bytes=large["expand_dim"] ** 2 * 4,
    )
    exact_matched = d3._projection_state_bytes(
        feature_dim=feature_dim, expand_dim=matched["expand_dim"], nonzeros=matched_nnz,
        num_classes=classes, gram_or_factor_bytes=matched["expand_dim"] ** 2 * 4,
    )
    srq = projected_srq_state_bytes(
        feature_dim=feature_dim, expand_dim=large["expand_dim"],
        synaptic_degree=large["synaptic_degree"], num_classes=classes,
        block_size=storage["block_size"], group_size=storage["group_size"],
    )["compressed_total_bytes"]
    config = {
        "schema_version": 1, "study_id": "synthetic-cub-d3",
        "dataset_identity": identity, "model_name": "tiny_backbone",
        "feature_dim": feature_dim, "checkpoint_sha256": "synthetic",
        "seed": 2025, "num_classes": classes, "num_tasks": tasks,
        "outer_validation_fraction": 0.2, "inner_validation_fraction": 0.2,
        "statistics_dtype": "float32", "solver_dtype": "float32",
        "raw_statistics_dtype": "float64",
        "selection_lambdas": [1.0, 10.0],
        "raw_selection_lambdas": [0.1, 1.0],
        "large_representation": large, "matched_representation": matched,
        "storage": storage,
        "expected_state": {
            "large_projection_nonzeros": large_nnz,
            "matched_projection_nonzeros": matched_nnz,
            "exact_large_bytes": exact_large, "srq_large_bytes": srq,
            "exact_matched_bytes": exact_matched,
        },
        "gates": {
            "maximum_solver_relative_residual": 10.0,
            "maximum_average_gap_to_exact_large_pp": 100.0,
            "maximum_final_gap_to_exact_large_pp": 100.0,
            "minimum_prediction_agreement": 0.0,
            "maximum_state_fraction_of_exact_large": 1.0,
            "maximum_state_mismatch_fraction": 0.999,
            "minimum_average_gain_over_state_matched_fly_pp": 0.0,
            "minimum_final_gain_over_state_matched_fly_pp": 0.0,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    args = argparse.Namespace(
        config=str(config_path), dataset_audit=str(audit),
        feature_cache_dir=str(cache), large_code_cache_dir=str(tmp_path / "large"),
        matched_code_cache_dir=str(tmp_path / "matched"),
        output_dir=str(tmp_path / "output"), device="cpu", require_test_hidden=True,
    )
    return args, config_path, cache


def test_locked_repository_config_and_state_accounting_are_valid():
    config = d3._read_config(Path("configs/srq_fly_cub_d3_train_only.json"))
    assert config["seed"] == 2025
    assert config["expected_state"]["srq_large_bytes"] == 105166628
    assert config["expected_state"]["exact_matched_bytes"] == 105149848


def test_projection_prefix_is_exact_but_does_not_claim_topk_transport():
    large = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    result = d3._verify_projection_prefix(large, large[:2].clone())
    assert result["verified"] is True
    assert "WTA Top-K is recomputed" in result["semantics"]
    with pytest.raises(ValueError, match="not the exact prefix"):
        d3._verify_projection_prefix(large, torch.ones(2, 2))


def test_selection_tie_break_and_result_state_contract():
    selected = d3._choose_candidate([
        {"ridge_lambda": 10.0, "validation_average_accuracy": 80.0},
        {"ridge_lambda": 1.0, "validation_average_accuracy": 80.0},
    ])
    assert selected["ridge_lambda"] == 1.0
    result = {
        "method": "exact", "status": "complete", "uses_test_set": False,
        "exemplar_free": True, "ridge_lambda": 1.0,
        "stage_accuracy": [70.0, 80.0], "validation_average_accuracy": 75.0,
        "persistent_state_bytes": 123, "maximum_solver_relative_residual": 1e-7,
    }
    d3._validate_result(
        result, name="exact", ridge_lambda=1.0, num_tasks=2,
        expected_state_bytes=123,
    )
    result["historical_features"] = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="invalid result contract"):
        d3._validate_result(
            result, name="exact", ridge_lambda=1.0, num_tasks=2,
            expected_state_bytes=123,
        )


def test_d3_synthetic_run_is_nested_train_only_and_resumable(tmp_path):
    args, _, cache = _write_fixture(tmp_path)
    first = d3.run(args)
    second = d3.run(args)
    assert first == second
    assert first["uses_test_set"] is False
    assert first["held_out_test_authorized"] is False
    assert first["uses_outer_validation_for_selection"] is False
    assert first["gates"]["projection_prefix_verified"] is True
    assert first["gates"]["runtime_state_matches_preregistered_accounting"] is True
    assert not (cache / "test.pt").exists()
    selection = json.loads((Path(args.output_dir) / "lambda_selection.json").read_text())
    assert selection["uses_outer_validation_for_selection"] is False
    assert "outer_validation_indices" not in selection


def test_d3_refuses_visible_test_and_invalid_seed(tmp_path):
    args, config_path, cache = _write_fixture(tmp_path)
    torch.save({"features": torch.zeros(1, 7)}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="held-out test.pt is visible"):
        d3.run(args)
    (cache / "test.pt").unlink()
    config = json.loads(config_path.read_text())
    config["seed"] = 1993
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="seed 2025"):
        d3._read_config(config_path)


def test_d3_rejects_duplicate_config_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate config key"):
        d3._read_config(path)
