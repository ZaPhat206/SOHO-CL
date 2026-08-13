import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import phasei_cub_selection


MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "phasei_cub_train_only_selection.json"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_audit(manifest):
    expected = manifest["dataset_identity"]
    return {
        "dataset": expected["dataset"],
        "dataset_identity_sha256": expected["dataset_identity_sha256"],
        "class_mapping_sha256": expected["class_mapping_sha256"],
        "cross_split_duplicate_content_count": 0,
        "train": {
            "image_count": expected["train_images"],
            "content_manifest_sha256": expected["train_content_manifest_sha256"],
        },
        "test": {
            "image_count": expected["test_images"],
            "content_manifest_sha256": expected["test_content_manifest_sha256"],
        },
    }


def cli(tmp_path, audit_path):
    feature_cache = tmp_path / "features"
    feature_cache.mkdir()
    (feature_cache / "train.pt").write_bytes(b"locked-train-cache")
    return SimpleNamespace(
        manifest=str(MANIFEST),
        manifest_sha256=file_hash(MANIFEST),
        dataset_audit=str(audit_path),
        feature_cache_dir=str(feature_cache),
        gate_cache_dir=str(tmp_path / "gate-cache"),
        output_dir=str(tmp_path / "selection"),
        device="cpu",
    )


def test_manifest_hash_mismatch_fails_before_dataset_or_features(tmp_path, monkeypatch):
    monkeypatch.setattr(
        phasei_cub_selection, "validate_dataset_audit",
        lambda *args: pytest.fail("dataset audit must not be reached"),
    )
    args = SimpleNamespace(
        manifest=str(MANIFEST), manifest_sha256="0" * 64,
        dataset_audit="missing", feature_cache_dir="missing",
        gate_cache_dir="missing", output_dir=str(tmp_path), device="cpu",
    )

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        phasei_cub_selection.run(args)


def test_locked_cub_protocol_uses_float64_statistics_after_numerical_preflight():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runtime = phasei_cub_selection.runtime_args(
        SimpleNamespace(
            feature_cache_dir="features", gate_cache_dir="gates",
            output_dir="output", device="cuda",
        ),
        manifest,
    )

    assert manifest["study_id"].endswith("_v2")
    assert runtime.statistics_dtype == "float64"


def test_dataset_identity_mismatch_fails_before_feature_cache(tmp_path, monkeypatch):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    audit = valid_audit(manifest)
    audit["dataset_identity_sha256"] = "wrong"
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(
        phasei_cub_selection, "validate_feature_metadata",
        lambda *args: pytest.fail("feature cache must not be reached"),
    )

    with pytest.raises(ValueError, match="dataset audit mismatch"):
        phasei_cub_selection.run(cli(tmp_path, audit_path))


def test_equal_budget_selection_is_train_only_and_resumable(tmp_path, monkeypatch):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(valid_audit(manifest)), encoding="utf-8")
    args = cli(tmp_path, audit_path)

    train = {
        "features": torch.zeros(manifest["dataset_identity"]["train_images"], 768),
        "labels": torch.zeros(manifest["dataset_identity"]["train_images"], dtype=torch.long),
    }
    metadata = {"synthetic_test_double": True}

    def fake_validate(runtime, _manifest):
        runtime.raw_dim = 768
        return train, metadata

    monkeypatch.setattr(phasei_cub_selection, "validate_feature_metadata", fake_validate)
    monkeypatch.setattr(phasei_cub_selection.crt_gate_runner, "prepare_cache", lambda runtime: {})
    monkeypatch.setattr(phasei_cub_selection.crt_gate_runner, "validate_gate_cache", lambda *args: {"test_cache_opened": False})
    monkeypatch.setattr(phasei_cub_selection.torch, "load", lambda *args, **kwargs: {})

    calls = {"raw": 0, "structured": 0}

    def result(method, score, candidate):
        return {
            **candidate,
            "method": method,
            "validation_average_incremental_accuracy": score,
            "solver_relative_residual_max": 1e-7,
            "uses_test_set": False,
        }

    def fake_raw(runtime, train_data, validation, snapshots, ridge):
        calls["raw"] += 1
        return result("raw_ridge", 8.0 + ridge, {"ridge_lambda": ridge})

    scores = {
        "anchor_only": 7.0,
        "full_raw_residual": 10.2,
        "schur_residual": 10.0,
        "fisher_residual": 9.0,
        "random_residual": 8.5,
    }

    def fake_structured(runtime, train_data, projection, validation, snapshots, candidate):
        calls["structured"] += 1
        return result(candidate["method"], scores[candidate["method"]], candidate)

    monkeypatch.setattr(phasei_cub_selection.crt_gate_runner, "_evaluate_raw_ridge", fake_raw)
    monkeypatch.setattr(phasei_cub_selection.crt_gate_runner, "_evaluate_candidate", fake_structured)

    first = phasei_cub_selection.run(args)
    first_calls = dict(calls)
    second = phasei_cub_selection.run(args)

    assert first["test_cache_opened"] is False
    assert first["held_out_test_authorized"] is True
    assert first["candidate_counts"] == {
        "raw_ridge": 4,
        "anchor_only": 4,
        "full_raw_residual": 36,
        "schur_residual": 27,
        "fisher_residual": 27,
        "random_residual": 27,
    }
    assert calls == first_calls  # Every second-run candidate came from the cache.
    assert second["source_train"] == first["source_train"]
    assert (Path(args.output_dir) / "selection_results.json").is_file()
