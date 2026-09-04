"""Correctness and protocol gates for SRQ-FLY Priority 4."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.flyhash import FlyHash
from tools import srq_fly_priority4_task_frequency as priority4


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_fly_priority4_cifar100_task_frequency_train_only.json"


def test_locked_priority4_config_is_valid_and_forbids_retuning(tmp_path):
    config = priority4._read_config(CONFIG)
    assert config["seed"] == 2025
    assert config["task_schedules"] == [10, 20]
    assert len(config["replicates"]) == 5
    assert config["hyperparameter_policy"]["retuning_allowed"] is False

    changed = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed["hyperparameter_policy"]["retuning_allowed"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="retuning"):
        priority4._read_config(path)


def test_schedule_independent_per_class_split_and_aligned_membership():
    labels = torch.arange(8).repeat_interleave(10)
    order = [5, 0, 7, 2, 4, 1, 6, 3]
    training, validation = priority4._per_class_split(
        labels, order, split_seed=8101, validation_fraction=0.2
    )
    train_two = priority4._group_parts(training, order, 2)
    train_four = priority4._group_parts(training, order, 4)
    val_two = priority4._group_parts(validation, order, 2)
    val_four = priority4._group_parts(validation, order, 4)
    assert torch.equal(
        torch.cat(train_two).sort().values, torch.cat(train_four).sort().values
    )
    assert torch.equal(
        torch.cat(val_two).sort().values, torch.cat(val_four).sort().values
    )
    assert not set(torch.cat(train_two).tolist()) & set(torch.cat(val_two).tolist())
    assert all(len(validation[class_id]) == 2 for class_id in order)


def _tiny_config():
    return {
        "feature_dim": 7, "num_classes": 8, "fly_ridge_lambda": 100.0,
        "representation": {
            "expand_dim": 18, "synaptic_degree": 3, "coding_level": 1 / 3,
            "encode_batch_size": 16, "evaluation_batch_size": 16,
        },
        "storage": {"block_size": 7, "group_size": 5},
        "p2b_backend": {
            "storage_mode": "int8", "update_backend": "blocked_qr",
            "update_panel_size": 128,
            "first_update_backend": "gram_cholesky",
            "quantization_backend": "streaming",
            "quantization_batch_blocks": 64,
        },
    }


def test_paired_evaluator_uses_identical_codes_and_is_exemplar_free():
    config = _tiny_config()
    generator = torch.Generator().manual_seed(99)
    labels = torch.arange(8).repeat_interleave(10)
    features = torch.randn(80, 7, generator=generator)
    class_order = list(range(8))
    training, validation = priority4._per_class_split(
        labels, class_order, split_seed=91, validation_fraction=0.2
    )
    training_parts = priority4._group_parts(training, class_order, 4)
    validation_parts = priority4._group_parts(validation, class_order, 4)
    active = 6
    code_indices = torch.stack([
        torch.randperm(18, generator=generator)[:active] for _ in range(80)
    ])
    code_values = torch.randn(80, active, generator=generator)
    torch.manual_seed(71)
    projection = FlyHash(7, 18, 3).projection_matrix.to_sparse_csc()
    result = priority4._evaluate_pair(
        config=config,
        train={"features": features, "labels": labels},
        code_cache=(code_indices, code_values, {}, projection),
        training_parts=training_parts, validation_parts=validation_parts,
        projection_seed=71, device=torch.device("cpu"),
    )
    assert result["status"] == "complete"
    assert result["uses_test_set"] is False
    assert len(result["task_diagnostics"]) == 4
    assert result["srq"]["persistent_state_bytes"] < result["exact"][
        "persistent_state_bytes"
    ]
    assert min(result["stage_prediction_agreement"]) >= 0
    assert len(result["final_exact_predictions"]) == 16


def test_replicate_comparison_uses_aligned_checkpoints_and_schedule_predictions():
    identity = {"id": 1, "class_order_seed": 1, "projection_seed": 2, "split_seed": 3}
    low = {
        "replicate": identity, "num_tasks": 2,
        "final_validation_indices_sha256": "same",
        "exact": {"stage_accuracy": [90.0, 80.0], "final_accuracy": 80.0},
        "srq": {"stage_accuracy": [89.0, 79.5], "final_accuracy": 79.5},
        "final_exact_predictions": [0, 1, 2, 3],
        "minimum_prediction_agreement": 0.98,
    }
    high = {
        "replicate": identity, "num_tasks": 4,
        "final_validation_indices_sha256": "same",
        "exact": {"stage_accuracy": [95.0, 90.0, 85.0, 80.0], "final_accuracy": 80.0},
        "srq": {"stage_accuracy": [94.0, 89.0, 84.0, 79.0], "final_accuracy": 79.0},
        "final_exact_predictions": [0, 1, 2, 3],
        "minimum_prediction_agreement": 0.97,
    }
    result = priority4._replicate_comparison(low, high)
    assert result["aligned_exact_aia_10"] == 85.0
    assert result["aligned_exact_aia_20"] == 85.0
    assert result["srq_exact_loss_10_pp"] == pytest.approx(0.75)
    assert result["srq_exact_loss_20_pp"] == pytest.approx(1.0)
    assert result["added_frequency_loss_pp"] == pytest.approx(0.25)
    assert result["exact_final_prediction_schedule_agreement"] == 1.0


def test_mismatched_final_validation_membership_is_rejected():
    identity = {"id": 1}
    low = {"replicate": identity, "num_tasks": 10,
           "final_validation_indices_sha256": "a"}
    high = {"replicate": identity, "num_tasks": 20,
            "final_validation_indices_sha256": "b"}
    with pytest.raises(ValueError, match="membership"):
        priority4._replicate_comparison(low, high)


def test_driver_locks_ten_paired_units_and_builds_aligned_summary(
    tmp_path, monkeypatch,
):
    config = priority4._read_config(CONFIG)
    source = priority4._source_identity()

    def fake_run(command, cwd):
        arguments = {command[index]: command[index + 1] for index in range(
            len(command) - 1
        ) if command[index].startswith("--")}
        replicate_id = int(arguments["--replicate"])
        num_tasks = int(arguments["--num-tasks"])
        replicate = config["replicates"][replicate_id - 1]
        exact_stage = [90.0] * num_tasks
        srq_stage = [89.9] * num_tasks if num_tasks == 10 else [89.8] * num_tasks
        payload = {
            "status": "complete", "uses_test_set": False,
            "replicate": replicate, "num_tasks": num_tasks,
            "config_sha256": priority4._sha256(CONFIG),
            "source_identity": source,
            "final_validation_indices_sha256": f"replicate-{replicate_id}",
            "exact": {
                "stage_accuracy": exact_stage, "final_accuracy": 90.0,
                "persistent_state_bytes": 400,
                "maximum_solver_relative_residual": 1e-7,
            },
            "srq": {
                "stage_accuracy": srq_stage,
                "final_accuracy": srq_stage[-1],
                "persistent_state_bytes": 90,
                "maximum_solver_relative_residual": 1e-7,
            },
            "minimum_prediction_agreement": 0.99,
            "final_exact_predictions": [0, 1, 2, 3],
            "final_srq_predictions": [0, 1, 2, 3],
        }
        Path(arguments["--output"]).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(priority4.subprocess, "run", fake_run)
    monkeypatch.setattr(
        priority4.subprocess, "check_output",
        lambda *args, **kwargs: "clean-commit\n"
        if "rev-parse" in args[0] else "",
    )
    feature_cache = tmp_path / "features"
    feature_cache.mkdir()
    result = priority4.run_driver(argparse.Namespace(
        config=str(CONFIG), feature_cache_dir=str(feature_cache),
        code_cache_root=str(tmp_path / "wta"),
        output_dir=str(tmp_path / "output"), device="cpu",
    ))
    assert result["status"] == "PASS_PRIORITY4_TASK_FREQUENCY"
    assert len(result["unit_files"]) == 10
    assert result["summaries"]["added_frequency_loss_pp"]["mean"] == pytest.approx(0.1)
    assert all(result["gates"].values())
