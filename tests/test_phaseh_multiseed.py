import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import torch

from tools import phaseh_multiseed


def _atomic_json(path, payload):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def test_repository_phaseh_manifest_is_locked_and_complete():
    path = Path("configs/phaseh_cifar100_multiseed.json")
    manifest, digest = phaseh_multiseed.load_locked_manifest(path)

    assert digest == phaseh_multiseed.EXPECTED_MANIFEST_SHA256
    assert tuple(manifest["methods"]) == phaseh_multiseed.METHODS
    assert manifest["shared_protocol"]["seeds"] == [1993, 2025, 3407, 4421, 5501]
    assert manifest["shared_protocol"]["test_time_hyperparameter_search"] is False


def test_manifest_byte_change_is_rejected(tmp_path):
    source = Path("configs/phaseh_cifar100_multiseed.json")
    changed = tmp_path / "changed.json"
    changed.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        phaseh_multiseed.load_locked_manifest(changed)


def test_phase_g_evidence_zip_identity_and_selected_config_are_verified(tmp_path):
    gate = b'{"locked": true}'
    gate_sha = hashlib.sha256(gate).hexdigest()
    source_gate_cache = {"source_train": {"bytes": 7, "sha256": "a" * 64}}
    lock = {
        "gate_results_sha256": gate_sha,
        "selected_proposal": {
            "method": "schur_residual", "rank": 2, "anchor_ridge": .1,
            "residual_ridge": .2, "complement_ridge": .3,
        },
        "selected_full_raw_residual": {"anchor_ridge": .1},
        "selected_raw_ridge": {"ridge_lambda": .4},
        "source_gate_cache": source_gate_cache,
    }
    result = {
        "lock": lock, "hyperparameter_search_performed": False,
        "test_cache_opened": True, "full_training_total_count": 50000,
        "environment": {"torch": "test"},
    }
    archive = tmp_path / "evidence.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("heldout_results.json", json.dumps(result))
        output.writestr("locked_manifest.json", json.dumps(lock))
        output.writestr("authorized_gate_results.json", gate)
    manifest = {
        "phase_g_evidence": {
            "heldout_zip_sha256": phaseh_multiseed.sha256(archive),
            "gate_results_sha256": gate_sha,
        },
        "methods": {
            "raw_ridge": {"ridge_lambda": .4},
            "full_raw_residual": {"anchor_ridge": .1},
            "schur_residual": {
                "rank": 2, "anchor_ridge": .1,
                "residual_ridge": .2, "complement_ridge": .3,
            },
        },
    }

    authorized = phaseh_multiseed.authorize_phase_g_evidence(archive, manifest)

    assert authorized["gate_results_sha256"] == gate_sha
    assert authorized["source_gate_cache"] == source_gate_cache
    manifest["phase_g_evidence"]["heldout_zip_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ZIP SHA-256 mismatch"):
        phaseh_multiseed.authorize_phase_g_evidence(archive, manifest)


def _synthetic_result(seed, method, offset=0.0):
    method_index = phaseh_multiseed.METHODS.index(method)
    value = float(seed % 100 + method_index + offset)
    return {
        "seed": seed, "method": method,
        "final_accuracy": value,
        "average_incremental_accuracy": value + 1,
        "forgetting": value / 10,
        "persistent_state_bytes": 1000 + method_index,
        "total_update_seconds": 1.0 + method_index,
        "total_inference_seconds": 2.0 + method_index,
        "peak_runtime_memory_bytes": 3000 + method_index,
        "exemplar_free": method != "soho_replay",
    }


def test_aggregate_reports_mean_std_paired_interval_and_soho_disclosure():
    seeds = [1993, 2025, 3407, 4421, 5501]
    manifest = {
        "shared_protocol": {"seeds": seeds},
        "reporting": {
            "paired_differences": ["schur_residual-raw_ridge"]
        },
    }
    results = [
        _synthetic_result(seed, method, offset=seed_index * .1)
        for seed_index, seed in enumerate(seeds)
        for method in phaseh_multiseed.METHODS
    ]

    aggregate = phaseh_multiseed._aggregate(results, manifest)

    summaries = {item["method"]: item for item in aggregate["method_summaries"]}
    assert summaries["flycl"]["average_incremental_accuracy_std"] > 0
    assert summaries["soho_replay"]["exemplar_free"] is False
    assert summaries["schur_residual"]["exemplar_free"] is True
    paired = aggregate["paired_differences"]
    assert len(paired) == 3
    assert all(len(item["confidence_interval_95"]) == 2 for item in paired)


def _runner_manifest():
    return {
        "shared_protocol": {
            "seeds": [1993, 2025, 3407, 4421, 5501],
            "num_classes": 100, "num_tasks": 10,
        },
        "stopping_rules": {
            "fly_reference_gate_seed": 1993,
            "fly_reference_gate_metric": "average_incremental_accuracy",
            "fly_reference_average_incremental_accuracy": 93.89,
            "fly_reference_tolerance_percentage_points": .5,
            "fly_reference_role": "external_paper_diagnostic_only",
            "stop_on_fly_discrepancy": False,
        },
        "reporting": {
            "paired_differences": ["schur_residual-raw_ridge"],
        },
    }


def test_fly_discrepancy_is_diagnostic_in_normal_study_order(tmp_path, monkeypatch):
    manifest = _runner_manifest()
    cache = {
        "features": torch.randn(100, 4),
        "labels": torch.arange(100),
    }
    calls = []

    monkeypatch.setattr(
        phaseh_multiseed, "load_locked_manifest",
        lambda path: (manifest, "t" * 64),
    )
    monkeypatch.setattr(
        phaseh_multiseed, "authorize_phase_g_evidence",
        lambda path, locked: {"zip_sha256": "t" * 64},
    )
    monkeypatch.setattr(phaseh_multiseed, "load_baseline_configs", lambda locked: {})
    monkeypatch.setattr(
        phaseh_multiseed, "validate_study_cache",
        lambda cache_dir, locked, evidence: (cache, cache, {"synthetic": True}),
    )
    monkeypatch.setattr(phaseh_multiseed, "sha256", lambda path: "t" * 64)

    def fake_evaluate(method, *args, **kwargs):
        calls.append(method)
        seed = args[2]
        result = _synthetic_result(seed, method)
        result["average_incremental_accuracy"] = 80.0
        matrix = [[80.0] * (stage + 1) for stage in range(10)]
        result.update(
            accuracy_matrix=matrix, accuracy_after_each_task=[80.0] * 10,
            peak_runtime_memory_bytes=None, diagnostics_by_task=[],
        )
        return result

    monkeypatch.setattr(phaseh_multiseed, "evaluate_method", fake_evaluate)
    args = type("Args", (), {
        "manifest": str(tmp_path / "manifest.json"),
        "phase_g_evidence_zip": str(tmp_path / "evidence.zip"),
        "feature_cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "output"), "device": "cpu",
    })()
    (tmp_path / "manifest.json").write_bytes(b"synthetic manifest")
    (tmp_path / "evidence.zip").write_bytes(b"synthetic evidence")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "train.pt").write_bytes(b"synthetic")

    result = phaseh_multiseed.run(args)

    assert result["status"] == "complete"
    assert calls[0] == "raw_ridge"
    assert calls[phaseh_multiseed.METHODS.index("flycl")] == "flycl"
    assert len(calls) == len(phaseh_multiseed.METHODS) * 5
    gate = json.loads((tmp_path / "output" / "fly_reference_gate.json").read_text())
    assert gate["within_reported_tolerance"] is False
    assert gate["stops_internal_study"] is False
    assert gate["role"] == "external_paper_diagnostic_only"
    assert not (tmp_path / "output" / "STOPPED_FLY_DISCREPANCY.json").exists()


def test_progress_line_is_concise(capsys):
    progress = phaseh_multiseed.Progress(40)
    progress.begin(1, 5, 2, 8, 1993, "flycl")
    progress.stage(1, 5, 2, 8, 4, 10, "flycl", "EVAL", "seen_tasks=4")
    progress.task(1, 5, 2, 8, 4, 10, progress.started)

    lines = capsys.readouterr().out.strip().splitlines()
    assert "start | seed 1/5=1993 | method 2/8=flycl" in lines[0]
    assert "stage=EVAL seen_tasks=4" in lines[1]
    assert "seed 1/5 | method 2/8 | task 4/10" in lines[2]
    assert "unit_eta=" in lines[2] and "study_eta=" in lines[2]
    assert all(len(line) < 180 for line in lines)


def test_corrupt_completed_result_is_rejected(tmp_path):
    path = tmp_path / "result.json"
    payload = _synthetic_result(1993, "raw_ridge")
    payload.update(
        completed=True,
        manifest_sha256="m" * 64,
        train_cache_sha256="t" * 64,
        class_order=list(range(100)),
        accuracy_matrix=[[1.0]],
        accuracy_after_each_task=[1.0],
    )
    _atomic_json(path, payload)

    with pytest.raises(ValueError, match="invalid resume result"):
        phaseh_multiseed._valid_completed_result(
            path, 1993, "raw_ridge", "m" * 64, "t" * 64
        )


@pytest.mark.parametrize("method", ["raw_ridge", "schur_residual"])
def test_real_method_evaluator_completes_tiny_global_stream(method):
    generator = torch.Generator().manual_seed(211)
    labels = torch.arange(4).repeat_interleave(6)
    train = {"features": torch.randn(24, 4, generator=generator), "labels": labels}
    test_labels = torch.arange(4).repeat_interleave(2)
    test = {"features": torch.randn(8, 4, generator=generator), "labels": test_labels}
    order = [2, 0, 3, 1]
    manifest = {
        "shared_protocol": {"seeds": [17], "num_classes": 4, "num_tasks": 2},
        "methods": {
            "raw_ridge": {"ridge_lambda": .1},
            "anchor_only": {
                "anchor_dim": 6, "synaptic_degree": 2,
                "coding_level": .5, "anchor_ridge": .1,
            },
            "schur_residual": {
                "rank": 2, "anchor_ridge": .1,
                "residual_ridge": .2, "complement_ridge": .3,
            },
        },
    }
    progress = phaseh_multiseed.Progress(1, device="cpu")

    result = phaseh_multiseed.evaluate_method(
        method, manifest, {}, 17, train, test,
        phaseh_multiseed.split(train["labels"], order, 2),
        phaseh_multiseed.split(test["labels"], order, 2),
        progress, 1, 1,
    )

    assert result["method"] == method
    assert result["exemplar_free"] is True
    assert len(result["accuracy_matrix"]) == 2
    assert all(len(row) == stage + 1 for stage, row in enumerate(result["accuracy_matrix"]))
    assert result["persistent_state_bytes"] > 0
