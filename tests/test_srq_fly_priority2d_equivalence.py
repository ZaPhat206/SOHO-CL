"""Locked protocol checks for real-data train-only backend equivalence."""

import argparse
import json
from pathlib import Path

import pytest
import torch

from tools import srq_fly_priority2d_equivalence as priority2d


ROOT = Path(__file__).resolve().parents[1]


def test_priority2d_config_is_strict_and_train_only():
    config = priority2d._read_config(
        ROOT / "configs/srq_fly_priority2d_cifar100_equivalence.json"
    )
    assert config["dataset"] == "CIFAR-100"
    assert config["num_tasks"] == 10
    assert config["quantization_batch_blocks"] == 64


def test_priority2d_constructs_only_first_update_backend_difference():
    config = priority2d._read_config(
        ROOT / "configs/srq_fly_priority2d_cifar100_equivalence.json"
    )
    projection = torch.zeros(
        config["representation"]["expand_dim"], config["feature_dim"]
    ).to_sparse_csc()
    baseline = priority2d._learner(
        config, "priority2b_batch64", projection, torch.device("cpu")
    )
    candidate = priority2d._learner(
        config, "implicit_ridge_batch64", projection, torch.device("cpu")
    )
    assert baseline.first_update_backend == "gram_cholesky"
    assert candidate.first_update_backend == "implicit_ridge_qr"
    assert baseline.quantization_backend == candidate.quantization_backend == "streaming"
    assert baseline.quantization_batch_blocks == candidate.quantization_batch_blocks == 64


def test_priority2d_refuses_visible_heldout_cache(tmp_path):
    cache = tmp_path / "features"
    cache.mkdir()
    torch.save({}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="test.pt"):
        priority2d.run_driver(argparse.Namespace(
            config=str(ROOT / "configs/srq_fly_priority2d_cifar100_equivalence.json"),
            feature_cache_dir=str(cache),
            code_cache_dir=str(tmp_path / "codes"),
            output_dir=str(tmp_path / "output"),
            device="cpu",
            require_clean_git=False,
        ))


def test_priority2d_unknown_config_field_is_rejected(tmp_path):
    payload = json.loads(
        (ROOT / "configs/srq_fly_priority2d_cifar100_equivalence.json").read_text()
    )
    payload["unknown"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="keys/schema"):
        priority2d._read_config(path)


def test_priority2d_workers_match_on_synthetic_train_only_stream(
    tmp_path, monkeypatch
):
    config = {
        "schema_version": 1,
        "study_id": "priority2d-test",
        "dataset": "Synthetic",
        "model_name": "tiny",
        "checkpoint_sha256": "synthetic",
        "feature_dim": 4,
        "seed": 2025,
        "num_classes": 4,
        "num_tasks": 2,
        "validation_fraction": 0.25,
        "statistics_dtype": "float32",
        "solver_dtype": "float32",
        "ridge_lambda": 100.0,
        "representation": {
            "expand_dim": 12,
            "synaptic_degree": 2,
            "coding_level": 0.25,
            "encode_batch_size": 8,
            "evaluation_batch_size": 8,
        },
        "storage": {"block_size": 4, "group_size": 3},
        "update_panel_size": 4,
        "quantization_batch_blocks": 64,
        "probe_rows": 4,
        "gates": {
            "maximum_stage_accuracy_gap_pp": 0.01,
            "maximum_relative_logit_drift": 1e-5,
            "maximum_solver_relative_residual": 2e-5,
            "maximum_peak_allocated_ratio_to_priority2b": 1.0,
            "maximum_update_ratio_to_priority2b": 1.25,
        },
    }
    generator = torch.Generator().manual_seed(77)
    labels = torch.tensor([0] * 6 + [1] * 6 + [2] * 6 + [3] * 6)
    train = {"features": torch.randn(24, 4, generator=generator), "labels": labels}
    code_indices = torch.stack([
        torch.randperm(12, generator=generator)[:3] for _ in range(24)
    ])
    code_values = torch.randn(24, 3, generator=generator)
    projection = torch.zeros(12, 4).to_sparse_csc()
    training = [torch.tensor(list(range(0, 5)) + list(range(6, 11))),
                torch.tensor(list(range(12, 17)) + list(range(18, 23)))]
    validation = [torch.tensor([5, 11]), torch.tensor([17, 23])]
    monkeypatch.setattr(priority2d, "_read_config", lambda _: config)
    monkeypatch.setattr(
        priority2d.p1,
        "_load_stream",
        lambda **kwargs: (
            train, [0, 1, 2, 3], training, validation,
            (code_indices, code_values, {}, projection),
        ),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    results = {}
    probes = {}
    for method in priority2d.METHODS:
        output = tmp_path / f"{method}.json"
        probe = tmp_path / f"{method}.pt"
        results[method] = priority2d.run_worker(argparse.Namespace(
            config=str(config_path),
            feature_cache_dir=str(tmp_path / "features"),
            code_cache_dir=str(tmp_path / "codes"),
            output=str(output),
            probe_output=str(probe),
            method=method,
            device="cpu",
        ))
        probes[method] = torch.load(probe, weights_only=True)
    assert results["priority2b_batch64"]["uses_test_set"] is False
    assert results["implicit_ridge_batch64"]["stage_accuracy"] == results[
        "priority2b_batch64"
    ]["stage_accuracy"]
    torch.testing.assert_close(
        probes["implicit_ridge_batch64"], probes["priority2b_batch64"],
        rtol=1e-5, atol=1e-7,
    )
