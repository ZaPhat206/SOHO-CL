import json
from pathlib import Path

import pytest
import torch

from tools import srq_fly_p2b_final_confirmation as confirmation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_fly_p2b_final_confirmation.json"
BASE_PROTOCOL = ROOT / "configs/srq_fly_selfcontained_final.json"


def test_locked_config_and_base_protocol_are_consistent():
    config = confirmation._read_config(CONFIG)
    protocol = confirmation._read_base_protocol(config, BASE_PROTOCOL)

    assert config["p2b_backend"]["first_update_backend"] == "gram_cholesky"
    assert config["p2b_backend"]["quantization_batch_blocks"] == 64
    assert len(protocol["final_evaluation"]["replicates"]) == 6
    assert config["final_evaluation"]["test_tuning_allowed"] is False
    assert "previously consumed" in config["final_evaluation"][
        "prior_test_use_disclosure"
    ]


def test_p2b_constructor_uses_selected_backend():
    config = confirmation._read_config(CONFIG)
    protocol = {
        "backbone": {"feature_dim": 4},
        "representation": {
            "expand_dim": 8,
            "synaptic_degree": 2,
            "coding_level": 0.25,
            "block_size": 4,
            "group_size": 2,
        },
    }
    projection = torch.eye(8, 4).to_sparse_csc()
    learner = confirmation._p2b_learner(
        config=config, protocol=protocol, ridge_lambda=10.0,
        projection=projection, seed=2025, device=torch.device("cpu"),
    )

    assert learner.update_backend == "blocked_qr"
    assert learner.first_update_backend == "gram_cholesky"
    assert learner.quantization_backend == "streaming"
    assert learner.quantization_batch_blocks == 64


def test_tiny_paired_confirmation_completes_without_sample_state():
    config = confirmation._read_config(CONFIG)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3] * 2)
    code_indices = torch.tensor([
        [row % 8, (row + 3) % 8] for row in range(len(labels))
    ])
    code_values = torch.tensor([
        [1.0 + 0.01 * row, 0.5 + 0.02 * row] for row in range(len(labels))
    ])
    projection = torch.eye(8, 4).to_sparse_csc()
    manifest = {
        "backbone": {"feature_dim": 4},
        "representation": {
            "large_expand_dim": 8,
            "synaptic_degree": 2,
            "coding_level": 0.25,
            "evaluation_batch_size": 4,
            "block_size": 4,
            "group_size": 2,
        },
    }
    stream = {"features": torch.zeros((len(labels), 4)), "labels": labels}
    result = confirmation._evaluate_paired_p2b(
        config=config, manifest=manifest,
        dataset={"fly_ridge_lambda": 10.0}, seed=2025, stream=stream,
        code_indices=code_indices, code_values=code_values,
        projection=projection,
        training_parts=[torch.arange(0, 4), torch.arange(4, 8)],
        test_parts=[torch.arange(8, 12), torch.arange(12, 16)],
        device=torch.device("cpu"),
    )

    assert result["status"] == "complete"
    assert result["exact"]["status"] == "complete"
    assert result["p2b"]["status"] == "complete"
    assert result["p2b"]["method"] == "srq_fly_p2b_10000"
    names = {row["name"].lower() for row in result["p2b"][
        "persistent_tensor_inventory"
    ]}
    assert not any("sample" in name or "history" in name for name in names)


def test_selection_evidence_declares_train_only_choices():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    evidence = config["selection_evidence"]

    assert evidence["artifact_sha256"] == (
        "e4b630781ff6f69deaecb63dda9926d256cd6b654ef4b51a682bf3ef94e6490b"
    )
    assert evidence["files"]["cifar100"]["fly_ridge_lambda"] == 1_000_000.0
    assert evidence["files"]["cub200"]["fly_ridge_lambda"] == 100_000.0
    assert evidence["files"]["imagenetr"]["raw_ridge_lambda"] == 1_000.0


def test_confirmation_authorization_detects_tampering(tmp_path, monkeypatch):
    config = confirmation._read_config(CONFIG)
    selections = {
        key: {"sha256": f"selection-{key}"} for key in confirmation.DATASET_KEYS
    }
    base_authorization = {"authorization_id": "base-lock"}
    record = {
        "schema_version": 1,
        "config_sha256": confirmation._sha256_file(CONFIG),
        "source_identity": confirmation._source_identity(),
        "base_authorization_id": "base-lock",
        "selection_sha256": {
            key: selections[key]["sha256"] for key in confirmation.DATASET_KEYS
        },
        "p2b_backend": config["p2b_backend"],
        "git_commit": "locked-commit",
        "git_dirty": False,
        "test_tuning_allowed": False,
        "accuracy_based_early_stop": False,
    }
    record["confirmation_authorization_id"] = confirmation._sha256_bytes(
        json.dumps(record, sort_keys=True).encode()
    )
    path = tmp_path / "confirmation_authorization.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    def fake_check_output(command, **_kwargs):
        return "locked-commit\n" if command[1:3] == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(confirmation.subprocess, "check_output", fake_check_output)
    validated = confirmation._validate_confirmation_authorization(
        path=path, config_path=CONFIG, config=config,
        base_authorization=base_authorization, selections=selections,
    )
    assert validated["confirmation_authorization_id"] == record[
        "confirmation_authorization_id"
    ]

    record["p2b_backend"]["quantization_batch_blocks"] = 16
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization mismatch"):
        confirmation._validate_confirmation_authorization(
            path=path, config_path=CONFIG, config=config,
            base_authorization=base_authorization, selections=selections,
        )
