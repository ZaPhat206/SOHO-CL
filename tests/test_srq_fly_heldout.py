import copy
import hashlib
import json
from pathlib import Path
import zipfile
import argparse

import pytest
import torch

from tools import srq_fly_heldout as heldout
from tools import srq_fly_extract_test as extract_test


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/srq_fly_three_dataset_heldout.json"


def _write_zip(path: Path, member: str, payload: dict) -> dict:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, json.dumps(payload))
    return {
        "artifact_name": path.name,
        "artifact_size": path.stat().st_size,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "result_member": member,
    }


def _synthetic_evidence(tmp_path: Path, *, uses_test=False):
    manifest = copy.deepcopy(heldout._read_manifest(MANIFEST_PATH))
    base = {"uses_test_set": uses_test, "held_out_test_authorized": False}
    payloads = {
        "cifar100": {
            **base, "status": "PASS_REVIEW_CIFAR_D5",
            "selected_fly_and_srq_lambda": 1e6, "fixed_raw_ridge_lambda": 0.01,
        },
        "cub200": {
            **base, "status": "STOP_SRQ_FLY_D4",
            "fixed_fly_ridge_lambda": 1e5, "selected_raw_ridge_lambda": 100.0,
        },
        "imagenetr": {
            **base, "status": "PASS_REVIEW_D21",
            "provenance": {"selected_lambda": 1e6},
        },
    }
    for key, payload in payloads.items():
        path = tmp_path / f"{key}.zip"
        evidence = manifest["datasets"][key]["train_only_evidence"]
        evidence.update(_write_zip(path, evidence["result_member"], payload))
    raw_path = tmp_path / "imagenetr_raw.zip"
    raw_evidence = manifest["datasets"]["imagenetr"]["raw_ridge_train_only_evidence"]
    raw_payload = {
        **base, "status": "STOP_SRQ_FLY_D1",
        "results": [{"method": "raw_ridge", "ridge_lambda": 0.01}],
    }
    raw_evidence.update(_write_zip(raw_path, raw_evidence["result_member"], raw_payload))
    return manifest


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def test_locked_manifest_validates_and_has_dataset_specific_tasks():
    manifest = heldout._read_manifest(MANIFEST_PATH)
    heldout._verify_method_identity(manifest)
    assert manifest["datasets"]["cifar100"]["num_tasks"] == 10
    assert manifest["datasets"]["cub200"]["num_tasks"] == 20
    assert manifest["datasets"]["imagenetr"]["num_tasks"] == 20
    assert manifest["reporting"]["test_tuning_allowed"] is False


def test_train_only_evidence_verifies_locked_lambdas(tmp_path):
    manifest = _synthetic_evidence(tmp_path)
    verified = heldout.verify_train_only_evidence(manifest, tmp_path)
    assert len(verified) == 4
    assert all(item["uses_test_set"] is False for item in verified)
    assert next(item for item in verified if item["dataset_key"] == "cifar100")["fly_ridge_lambda"] == 1e6


def test_train_only_evidence_rejects_test_use(tmp_path):
    manifest = _synthetic_evidence(tmp_path, uses_test=True)
    with pytest.raises(ValueError, match="train-only result contract"):
        heldout.verify_train_only_evidence(manifest, tmp_path)


def test_authorization_is_idempotent_only_for_same_context(tmp_path):
    manifest = _synthetic_evidence(tmp_path)
    manifest_path = _write_manifest(tmp_path, manifest)
    first = heldout.authorize(manifest_path, tmp_path, tmp_path / "out", False)
    second = heldout.authorize(manifest_path, tmp_path, tmp_path / "out", False)
    assert first["authorization_id"] == second["authorization_id"]
    changed = copy.deepcopy(manifest)
    changed["datasets"]["cifar100"]["raw_ridge_lambda"] = 0.02
    changed_path = _write_manifest(tmp_path, changed)
    with pytest.raises(ValueError, match="locked raw-Ridge lambda mismatch"):
        heldout.authorize(changed_path, tmp_path, tmp_path / "out", False)


def test_metric_definitions_include_valid_forgetting():
    metrics = heldout._result_metrics([[90.0], [80.0, 70.0], [75.0, 65.0, 60.0]])
    assert metrics["stage_accuracy"] == [90.0, 75.0, pytest.approx(200 / 3)]
    assert metrics["final_accuracy"] == pytest.approx(200 / 3)
    assert metrics["average_incremental_accuracy"] == pytest.approx((90 + 75 + 200 / 3) / 3)
    assert metrics["forgetting"] == pytest.approx(10.0)


def test_sample_state_inventory_rejects_forbidden_or_sample_shaped_tensor():
    with pytest.raises(AssertionError, match="forbidden"):
        heldout._assert_sample_free_inventory(
            [{"name": "historical_features", "shape": [8, 3]}], 8, {3}
        )
    with pytest.raises(AssertionError, match="historical sample dimension"):
        heldout._assert_sample_free_inventory(
            [{"name": "mystery", "shape": [8, 2]}], 8, {2, 3}
        )
    heldout._assert_sample_free_inventory(
        [{"name": "Q", "shape": [8, 2]}], 8, {8, 2, 3}
    )


def test_runtime_state_contract_uses_realized_sparse_entries():
    manifest = copy.deepcopy(heldout._read_manifest(MANIFEST_PATH))
    manifest["backbone"]["feature_dim"] = 3
    manifest["representation"]["large_expand_dim"] = 4
    manifest["representation"]["synaptic_degree"] = 2
    dataset = manifest["datasets"]["cifar100"]
    dataset.update({
        "num_classes": 2, "matched_expand_dim": 2,
        "nominal_exact_large_bytes": 1000, "nominal_srq_large_bytes": 500,
        "nominal_exact_matched_bytes": 300, "maximum_missing_projection_entries": 2,
    })
    large = torch.tensor([[1., 1., 0.], [1., 0., 1.], [0., 1., 1.], [1., 0., 1.]]).to_sparse_csc()
    matched = torch.tensor([[1., 1., 0.], [1., 0., 1.]]).to_sparse_csc()
    contract = heldout._runtime_state_contract(manifest, dataset, large, matched)
    assert contract["large_missing_projection_entries"] == 0
    assert contract["matched_missing_projection_entries"] == 0
    assert contract["raw_ridge_bytes"] == (9 + 12 + 2) * 8


def _tiny_stream():
    torch.manual_seed(11)
    train_x = torch.randn(8, 3)
    test_x = torch.randn(4, 3)
    train_y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    test_y = torch.tensor([0, 0, 1, 1])
    stream = {"features": torch.cat((train_x, test_x)), "labels": torch.cat((train_y, test_y))}
    training = [torch.arange(0, 4), torch.arange(4, 8)]
    testing = [torch.arange(8, 10), torch.arange(10, 12)]
    indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [0, 5], [2, 4], [1, 3], [0, 2], [2, 5], [1, 4], [3, 5]], dtype=torch.int16)
    values = torch.randn(12, 2)
    projection = torch.randn(6, 3).to_sparse_csc()
    return stream, training, testing, indices, values, projection


def test_synthetic_paired_heldout_is_task_id_free_and_sample_free():
    manifest = copy.deepcopy(heldout._read_manifest(MANIFEST_PATH))
    manifest["backbone"]["feature_dim"] = 3
    manifest["representation"].update({
        "large_expand_dim": 6, "synaptic_degree": 2, "coding_level": 1 / 3,
        "encode_batch_size": 4, "evaluation_batch_size": 2,
        "block_size": 3, "group_size": 2,
    })
    dataset = copy.deepcopy(manifest["datasets"]["cifar100"])
    dataset.update({"num_classes": 2, "num_tasks": 2, "fly_ridge_lambda": 10.0})
    stream, training, testing, indices, values, projection = _tiny_stream()
    result = heldout._evaluate_paired(
        manifest=manifest, dataset=dataset, seed=2025, stream=stream,
        code_indices=indices, code_values=values, projection=projection,
        training_parts=training, test_parts=testing, device=torch.device("cpu"),
    )
    assert result["status"] == "complete" and result["uses_test_set"] is True
    assert result["srq"]["exemplar_free"] is True
    assert len(result["srq"]["accuracy_matrix"]) == 2
    names = {item["name"] for item in result["srq"]["persistent_tensor_inventory"]}
    assert not any("sample" in name or "feature" in name for name in names)


def test_synthetic_matched_and_raw_heldout_metrics_are_complete():
    manifest = copy.deepcopy(heldout._read_manifest(MANIFEST_PATH))
    manifest["backbone"]["feature_dim"] = 3
    manifest["representation"].update({
        "large_expand_dim": 6, "synaptic_degree": 2, "coding_level": 1 / 3,
        "encode_batch_size": 4, "evaluation_batch_size": 2,
    })
    dataset = copy.deepcopy(manifest["datasets"]["cifar100"])
    dataset.update({
        "num_classes": 2, "num_tasks": 2, "matched_expand_dim": 4,
        "fly_ridge_lambda": 10.0, "raw_ridge_lambda": 1.0,
    })
    stream, training, testing, _, _, _ = _tiny_stream()
    indices = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]] * 3, dtype=torch.int16)
    values = torch.randn(12, 2)
    projection = torch.randn(4, 3).to_sparse_csc()
    matched = heldout._evaluate_exact_matched(
        manifest=manifest, dataset=dataset, seed=2025, stream=stream,
        code_indices=indices, code_values=values, projection=projection,
        training_parts=training, test_parts=testing, device=torch.device("cpu"),
    )
    raw = heldout._evaluate_raw(
        manifest=manifest, dataset=dataset, seed=2025, stream=stream,
        training_parts=training, test_parts=testing, device=torch.device("cpu"),
    )
    for result in (matched, raw):
        assert result["status"] == "complete" and result["uses_test_set"] is True
        assert len(result["accuracy_matrix"]) == 2
        assert result["total_update_seconds"] >= 0
        assert result["total_inference_seconds"] >= 0


def test_summary_reports_without_accuracy_gate(tmp_path):
    manifest = heldout._read_manifest(MANIFEST_PATH)
    manifest_path = _write_manifest(tmp_path, manifest)
    for dataset_key in manifest["datasets"]:
        seed_results = []
        for seed in manifest["seeds"]:
            methods = {
                method: {
                    "status": "complete", "final_accuracy": 80.0,
                    "average_incremental_accuracy": 85.0 + (method == "srq_fly_10000"),
                    "forgetting": 3.0, "persistent_state_bytes": 100,
                    "total_update_seconds": 1.0, "total_inference_seconds": 2.0,
                }
                for method in heldout.METHODS
            }
            seed_results.append({"seed": seed, "methods": methods})
        path = tmp_path / dataset_key / "heldout_results.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "uses_test_set": True, "test_tuning_allowed": False,
            "seed_results": seed_results,
        }), encoding="utf-8")
    summary = heldout.summarize(manifest_path, tmp_path)
    assert summary["status"] == "REPORTED_WITHOUT_ACCURACY_GATE"
    assert summary["paired_srq_minus_state_matched_fly"]["cifar100"]["mean"] == 1.0


def test_test_extractor_restores_only_under_matching_authorization(tmp_path):
    manifest = _synthetic_evidence(tmp_path)
    manifest["datasets"]["cifar100"].update({"train_samples": 4, "test_samples": 2})
    manifest_path = _write_manifest(tmp_path, manifest)
    authorization = heldout.authorize(manifest_path, tmp_path, tmp_path / "out", False)
    cache = tmp_path / "cache"; cache.mkdir()
    (cache / "metadata.json").write_text(json.dumps({
        "dataset": "CIFAR-100", "backbone_model": "vit_base_patch16_224",
        "checkpoint_sha256": manifest["backbone"]["checkpoint_sha256"],
        "preprocessing": "vit", "test_features_materialized": True,
    }), encoding="utf-8")
    torch.save({"features": torch.randn(4, 768), "labels": torch.tensor([0, 0, 1, 1])}, cache / "train.pt")
    torch.save({"features": torch.randn(2, 768), "labels": torch.tensor([0, 1])}, cache / "test.pt")
    result = extract_test.extract_test(argparse.Namespace(
        manifest=str(manifest_path), dataset_key="cifar100",
        authorization=str(tmp_path / "out" / "heldout_authorization.json"),
        feature_cache_dir=str(cache), root="unused", backbone_checkpoint="unused",
        device="cpu", batch_size=2, num_workers=0,
    ))
    assert result["status"] == "restored"
    assert authorization["authorization_id"]


def test_final_notebook_code_cells_compile_and_bind_hashes():
    notebook_path = ROOT / "notebooks/srq_fly_final_three_dataset_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{index}", "exec")
    text = notebook_path.read_text(encoding="utf-8")
    for path in (
        ROOT / "configs/srq_fly_three_dataset_heldout.json",
        ROOT / "tools/srq_fly_heldout.py",
        ROOT / "tools/srq_fly_extract_test.py",
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text
