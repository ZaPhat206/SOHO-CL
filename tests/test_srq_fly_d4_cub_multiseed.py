"""Synthetic tests for prospective SRQ-FLY D4 multi-seed confirmation."""

import argparse
import json
from pathlib import Path

import pytest
import torch

from methods.srq_fly import projected_srq_state_bytes
from tools import srq_fly_d3_cub as d3
from tools import srq_fly_d4_cub_multiseed as d4
from tools.twa_fly_pilot import _sha256_file


def _fixture(tmp_path: Path):
    feature_dim, classes = 7, 200
    generator = torch.Generator().manual_seed(818)
    features, labels = [], []
    for class_id in range(classes):
        center = torch.randn(feature_dim, generator=generator) + class_id / 60
        features.append(center + 0.12 * torch.randn(6, feature_dim, generator=generator))
        labels.append(torch.full((6,), class_id, dtype=torch.long))
    train = {"features": torch.cat(features), "labels": torch.cat(labels)}
    cache = tmp_path / "features"
    cache.mkdir()
    torch.save(train, cache / "train.pt")
    (cache / "metadata.json").write_text(json.dumps({
        "schema_version": 1, "dataset": "CUB-200-2011",
        "dataset_version": "processed-imagefolder", "backbone_model": "tiny",
        "checkpoint_sha256": "synthetic", "preprocessing": "vit",
        "feature_dim": feature_dim, "finite": True,
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
        "dataset": identity["dataset"],
        "dataset_identity_sha256": identity["dataset_identity_sha256"],
        "class_mapping_sha256": identity["class_mapping_sha256"],
        "cross_split_duplicate_content_count": 0,
        "train": {"image_count": identity["train_samples"], "content_manifest_sha256": identity["train_content_manifest_sha256"]},
        "test": {"image_count": identity["test_samples"], "content_manifest_sha256": identity["test_content_manifest_sha256"]},
    }), encoding="utf-8")
    train_sha = _sha256_file(cache / "train.pt")
    d3_payload = {
        "status": "STOP_SRQ_FLY_D3", "uses_test_set": False,
        "held_out_test_authorized": False,
        "uses_outer_validation_for_selection": False,
        "selection": {"exact_fly_10000_and_srq_10000": 1.0},
        "provenance": {
            "config_sha256": "d3-config", "runner_git_commit": "d3-commit",
            "runner_git_dirty": False, "train_sha256": train_sha,
        },
        "gates": {"numerical_stability": False, "accuracy": True, "state": True},
        "results": [
            {
                "method": "srq_int8", "validation_average_accuracy": 81.0,
                "stage_accuracy": [80.0, 82.0], "persistent_state_bytes": 1000,
            },
            {
                "method": "exact_fly_4518", "validation_average_accuracy": 80.0,
                "stage_accuracy": [79.0, 81.0], "persistent_state_bytes": 999,
            },
        ],
    }
    d3_path = tmp_path / "d3_results.json"
    d3_path.write_text(json.dumps(d3_payload), encoding="utf-8")
    large = {
        "expand_dim": 12, "synaptic_degree": 3, "coding_level": 0.25,
        "encode_batch_size": 128, "evaluation_batch_size": 128,
    }
    matched = {
        "expand_dim": 6, "synaptic_degree": 3, "coding_level": 0.25,
        "encode_batch_size": 128, "evaluation_batch_size": 128,
    }
    storage = {"block_size": 4, "group_size": 2}
    large_entries = large["expand_dim"] * large["synaptic_degree"]
    matched_entries = matched["expand_dim"] * matched["synaptic_degree"]
    exact_large = d3._projection_state_bytes(
        feature_dim=feature_dim, expand_dim=large["expand_dim"],
        nonzeros=large_entries, num_classes=classes,
        gram_or_factor_bytes=large["expand_dim"] ** 2 * 4,
    )
    exact_matched = d3._projection_state_bytes(
        feature_dim=feature_dim, expand_dim=matched["expand_dim"],
        nonzeros=matched_entries, num_classes=classes,
        gram_or_factor_bytes=matched["expand_dim"] ** 2 * 4,
    )
    srq = projected_srq_state_bytes(
        feature_dim=feature_dim, expand_dim=large["expand_dim"],
        synaptic_degree=large["synaptic_degree"], num_classes=classes,
        block_size=storage["block_size"], group_size=storage["group_size"],
    )["compressed_total_bytes"]
    reference = {
        "artifact_sha256": "synthetic-zip",
        "result_sha256": _sha256_file(d3_path), "config_sha256": "d3-config",
        "runner_git_commit": "d3-commit", "train_sha256": train_sha,
        "status": "STOP_SRQ_FLY_D3", "selected_fly_ridge_lambda": 1.0,
        "srq_average_accuracy": 81.0, "srq_final_accuracy": 82.0,
        "srq_persistent_state_bytes": 1000,
        "matched_average_accuracy": 80.0, "matched_final_accuracy": 81.0,
        "matched_persistent_state_bytes": 999,
    }
    config = {
        "schema_version": 1, "study_id": "synthetic-d4",
        "dataset_identity": identity, "model_name": "tiny",
        "feature_dim": feature_dim, "checkpoint_sha256": "synthetic",
        "seeds": [2026, 2027, 2028, 2029, 2030],
        "num_classes": classes, "num_tasks": 20,
        "outer_validation_fraction": 0.2, "inner_validation_fraction": 0.2,
        "statistics_dtype": "float32", "solver_dtype": "float32",
        "raw_statistics_dtype": "float64", "fixed_fly_ridge_lambda": 1.0,
        "raw_selection_lambdas": [0.1, 10.0, 100.0],
        "large_representation": large, "matched_representation": matched,
        "storage": storage,
        "expected_state": {
            "nominal_large_projection_entries": large_entries,
            "nominal_matched_projection_entries": matched_entries,
            "maximum_missing_projection_entries": 10,
            "nominal_exact_large_bytes": exact_large,
            "nominal_srq_large_bytes": srq,
            "nominal_exact_matched_bytes": exact_matched,
        },
        "reference_d3": reference,
        "gates": {
            "maximum_search_candidate_solver_relative_residual": 10.0,
            "maximum_outer_solver_relative_residual": 10.0,
            "maximum_average_gap_to_exact_large_pp": 100.0,
            "maximum_final_gap_to_exact_large_pp": 100.0,
            "minimum_prediction_agreement": 0.0,
            "maximum_state_fraction_of_exact_large": 1.0,
            "maximum_state_mismatch_fraction": 0.999,
            "minimum_mean_average_gain_over_state_matched_fly_pp": 0.0,
            "minimum_median_average_gain_over_state_matched_fly_pp": 0.0,
            "minimum_mean_final_gain_over_state_matched_fly_pp": 0.0,
            "minimum_seed_win_fraction": 0.2,
            "maximum_worst_seed_average_loss_pp": 100.0,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    args = argparse.Namespace(
        config=str(config_path), dataset_audit=str(audit),
        feature_cache_dir=str(cache), d3_result=str(d3_path),
        code_cache_root=str(tmp_path / "codes"), output_dir=str(tmp_path / "out"),
        device="cpu", require_test_hidden=True,
    )
    return args, config_path, cache, d3_path


def test_locked_d4_config_is_valid_and_uses_fresh_seeds():
    config = d4._read_config(Path("configs/srq_fly_cub_d4_multiseed_train_only.json"))
    assert config["seeds"] == [2026, 2027, 2028, 2029, 2030]
    assert config["fixed_fly_ridge_lambda"] == 100000.0
    assert max(config["raw_selection_lambdas"]) > 10


def test_summary_reports_sample_std_and_five_seed_t_interval():
    summary = d4._mean_std_ci([1, 2, 3, 4, 5])
    assert summary["mean"] == 3
    assert summary["sample_std"] == pytest.approx(2.5 ** 0.5)
    assert summary["ci95_low"] < 3 < summary["ci95_high"]


def test_d3_reference_requires_numerical_only_failure(tmp_path):
    args, config_path, _, d3_path = _fixture(tmp_path)
    config = d4._read_config(config_path)
    d4._verify_d3_reference(d3_path, config, config["reference_d3"]["train_sha256"])
    payload = json.loads(d3_path.read_text())
    payload["gates"]["accuracy"] = False
    d3_path.write_text(json.dumps(payload))
    config["reference_d3"]["result_sha256"] = _sha256_file(d3_path)
    with pytest.raises(ValueError, match="solely"):
        d4._verify_d3_reference(
            d3_path, config, config["reference_d3"]["train_sha256"]
        )


def test_d4_synthetic_run_is_five_seed_train_only_and_resumable(tmp_path):
    args, _, cache, _ = _fixture(tmp_path)
    first = d4.run(args)
    second = d4.run(args)
    assert first == second
    assert len(first["seed_results"]) == 5
    assert [item["seed"] for item in first["seed_results"]] == [2026, 2027, 2028, 2029, 2030]
    assert first["uses_test_set"] is False
    assert first["held_out_test_authorized"] is False
    assert first["uses_outer_validation_for_selection"] is False
    assert not (cache / "test.pt").exists()
    selection = json.loads(
        (Path(args.output_dir) / "raw_lambda_selection.json").read_text()
    )
    assert selection["uses_outer_validation_for_selection"] is False
    assert len(selection["per_seed_candidates"]) == 15
    assert (Path(args.output_dir) / "d4_results.json").is_file()


def test_d4_refuses_visible_test_and_tampered_reference(tmp_path):
    args, config_path, cache, d3_path = _fixture(tmp_path)
    torch.save({"features": torch.zeros(1, 7)}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="held-out test.pt is visible"):
        d4.run(args)
    (cache / "test.pt").unlink()
    payload = json.loads(d3_path.read_text())
    payload["results"][0]["validation_average_accuracy"] = 82.0
    d3_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        d4.run(args)
    config = json.loads(config_path.read_text())
    config["seeds"][-1] = 2031
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fresh preregistered seeds"):
        d4._read_config(config_path)


def test_d4_rejects_duplicate_config_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate config key"):
        d4._read_config(path)
