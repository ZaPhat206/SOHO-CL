"""Correctness gates for the repaired direct-quantization control."""

import argparse
import json
from pathlib import Path

import pytest
import torch

from methods.srq_fly_optimized.direct_control import (
    CertifiedDirectInt8GramLearner,
    _quantization_error_metrics,
)
from tools import srq_fly_priority3_direct_control as priority3


def _kwargs():
    return {
        "feature_dim": 7,
        "expand_dim": 18,
        "synaptic_degree": 3,
        "coding_level": 1 / 3,
        "ridge_lambda": 100.0,
        "block_size": 7,
        "group_size": 5,
        "seed": 2025,
        "device": "cpu",
        "statistics_dtype": torch.float64,
        "solver_dtype": torch.float64,
        "repair_margin_multiplier": 8.0,
        "repair_error_chunk_size": 4,
    }


def _stream():
    generator = torch.Generator().manual_seed(8102)
    first = torch.randn(17, 18, generator=generator, dtype=torch.float64)
    second = 1.7 * torch.randn(13, 18, generator=generator, dtype=torch.float64)
    first_labels = torch.arange(17) % 3
    second_labels = 3 + torch.arange(13) % 3
    return first, first_labels, second, second_labels


def test_error_metric_bounds_symmetric_spectral_error():
    generator = torch.Generator().manual_seed(19)
    reference = torch.randn(11, 11, generator=generator, dtype=torch.float64)
    reference = (reference + reference.T) * 0.5
    perturbation = 0.01 * torch.randn(
        11, 11, generator=generator, dtype=torch.float64
    )
    perturbation = (perturbation + perturbation.T) * 0.5
    infinity_bound, relative = _quantization_error_metrics(
        reference + perturbation, reference, row_chunk_size=3
    )
    spectral = float(torch.linalg.matrix_norm(perturbation, ord=2))
    assert infinity_bound + 1e-12 >= spectral
    assert relative > 0


def test_certified_direct_gram_is_spd_and_exemplar_free_after_each_task():
    learner = CertifiedDirectInt8GramLearner(**_kwargs())
    first, first_labels, second, second_labels = _stream()
    for codes, labels in ((first, first_labels), (second, second_labels)):
        learner.update_codes(codes, labels)
        learner.assert_exemplar_free_state()
        reconstructed = learner.gram.reconstruct_symmetric(dtype=torch.float64)
        system = reconstructed + (
            learner.ridge_lambda + float(learner.diagonal_loading)
        ) * torch.eye(learner.expand_dim, dtype=torch.float64)
        assert float(torch.linalg.eigvalsh(system).amin()) > 0
        assert learner.diagnostics["certified_system_eigenvalue_floor"] > 0
        assert learner.diagnostics["effective_ridge_lambda"] >= learner.ridge_lambda
        assert learner.diagnostics["solver_relative_residual"] < 1e-9
    inventory = learner.persistent_tensors()
    assert inventory["certified_gram_lower_bound"].shape == ()
    assert inventory["diagonal_loading"].shape == ()
    assert not any("sample" in name or "history" in name for name in inventory)


def test_certified_direct_gram_checkpoint_roundtrip_and_config_lock():
    first, first_labels, second, second_labels = _stream()
    learner = CertifiedDirectInt8GramLearner(**_kwargs())
    learner.update_codes(first, first_labels)
    state = learner.state_dict()

    resumed = CertifiedDirectInt8GramLearner(**_kwargs())
    resumed.load_state_dict(state)
    torch.testing.assert_close(resumed.weights, learner.weights, rtol=0, atol=0)
    assert resumed.persistent_state_bytes() == learner.persistent_state_bytes()
    resumed.update_codes(second, second_labels)
    assert resumed.diagnostics["certified_system_eigenvalue_floor"] > 0

    changed = _kwargs()
    changed["repair_margin_multiplier"] = 4.0
    with pytest.raises(ValueError, match="margin multiplier"):
        CertifiedDirectInt8GramLearner(**changed).load_state_dict(state)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("repair_margin_multiplier", 0.0, "margin"),
        ("repair_error_chunk_size", 0, "chunk"),
    ],
)
def test_repair_configuration_validation(field, value, message):
    kwargs = _kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        CertifiedDirectInt8GramLearner(**kwargs)


def _protocol():
    return {
        "schema_version": 1,
        "study_id": "priority3-synthetic",
        "dataset": "CIFAR-100",
        "model_name": "tiny_backbone",
        "checkpoint_sha256": "synthetic-checkpoint",
        "feature_dim": 7,
        "seed": 2025,
        "num_classes": 6,
        "num_tasks": 3,
        "validation_fraction": 0.2,
        "statistics_dtype": "float32",
        "solver_dtype": "float32",
        "fly_ridge_lambda": 100.0,
        "representation": {
            "expand_dim": 18,
            "synaptic_degree": 3,
            "coding_level": 1 / 3,
            "encode_batch_size": 16,
            "evaluation_batch_size": 16,
        },
        "storage": {"block_size": 7, "group_size": 5},
        "p2b_backend": {
            "update_backend": "blocked_qr",
            "update_panel_size": 128,
            "first_update_backend": "gram_cholesky",
            "quantization_backend": "streaming",
            "quantization_batch_blocks": 64,
        },
        "direct_gram_repair": {
            "name": "weyl_infinity_norm_diagonal_loading",
            "margin_multiplier": 8.0,
            "error_chunk_size": 4,
            "uses_labels_or_accuracy": False,
            "adaptive_retry_allowed": False,
        },
        "gates": {
            "maximum_solver_relative_residual": 1e-3,
            "practical_equivalence_pp": 0.1,
            "material_square_root_advantage_pp": 0.5,
        },
    }


def _feature_fixture(tmp_path: Path):
    cache = tmp_path / "features"
    cache.mkdir()
    generator = torch.Generator().manual_seed(721)
    features, labels = [], []
    for class_id in range(6):
        center = torch.randn(7, generator=generator) + class_id / 2
        features.append(center + 0.1 * torch.randn(10, 7, generator=generator))
        labels.append(torch.full((10,), class_id, dtype=torch.long))
    torch.save(
        {"features": torch.cat(features), "labels": torch.cat(labels)},
        cache / "train.pt",
    )
    (cache / "metadata.json").write_text(
        json.dumps({
            "schema_version": 1,
            "dataset": "CIFAR-100",
            "backbone_model": "tiny_backbone",
            "checkpoint_sha256": "synthetic-checkpoint",
            "feature_dim": 7,
            "finite": True,
            "test_features_materialized": False,
        }),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(_protocol()), encoding="utf-8")
    return cache, protocol


def test_priority3_worker_is_train_only_and_exports_repair_certificate(tmp_path):
    cache, protocol = _feature_fixture(tmp_path)
    output = tmp_path / "repair.json"
    result = priority3.run_worker(argparse.Namespace(
        config=str(protocol),
        feature_cache_dir=str(cache),
        code_cache_dir=str(tmp_path / "wta"),
        method="direct_int8_gram_weyl_repair",
        output=str(output),
        device="cpu",
    ))
    assert result["status"] == "complete"
    assert result["uses_test_set"] is False
    assert len(result["task_diagnostics"]) == 3
    assert all(
        row["certified_system_eigenvalue_floor"] > 0
        for row in result["task_diagnostics"]
    )
    assert output.is_file() and not (cache / "test.pt").exists()


def test_priority3_protocol_forbids_seed_change_accuracy_repair_and_visible_test(tmp_path):
    cache, protocol = _feature_fixture(tmp_path)
    payload = _protocol()
    payload["seed"] = 1993
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="2025"):
        priority3._read_config(protocol)
    payload = _protocol()
    payload["direct_gram_repair"]["uses_labels_or_accuracy"] = True
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="repair contract"):
        priority3._read_config(protocol)

    protocol.write_text(json.dumps(_protocol()), encoding="utf-8")
    torch.save({}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="test.pt"):
        priority3.run_worker(argparse.Namespace(
            config=str(protocol),
            feature_cache_dir=str(cache),
            code_cache_dir=str(tmp_path / "wta"),
            method="exact_fly_10000",
            output=str(tmp_path / "exact.json"),
            device="cpu",
        ))


def test_priority3_interpretation_thresholds_are_locked():
    config = _protocol()
    exact = {"validation_average_accuracy": 90.0}
    repaired = {"validation_average_accuracy": 89.0}
    close = {"validation_average_accuracy": 89.05}
    material = {"validation_average_accuracy": 89.6}
    assert "LOW_BITS_SUFFICIENT" in priority3._interpret(
        config, exact, repaired, close
    )
    assert "MATERIAL_ACCURACY_ADVANTAGE" in priority3._interpret(
        config, exact, repaired, material
    )


def test_priority3_isolated_driver_completes_all_controls(tmp_path):
    cache, protocol = _feature_fixture(tmp_path)
    output = tmp_path / "output"
    result = priority3.run_driver(argparse.Namespace(
        config=str(protocol),
        feature_cache_dir=str(cache),
        code_cache_dir=str(tmp_path / "wta"),
        output_dir=str(output),
        device="cpu",
    ))
    assert result["status"] == "COMPLETE_REVIEW_PRIORITY3"
    assert result["uses_test_set"] is False
    assert len(result["results"]) == len(priority3.METHODS)
    assert all(result["methodological_gates"].values())
    assert (output / "priority3_results.json").is_file()
