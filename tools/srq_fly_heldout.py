"""Locked three-dataset held-out evaluator for SRQ-FLY.

The command has no hyperparameter arguments.  Every method setting comes from
the reviewed manifest.  Sample-level feature/WTA caches are experiment
infrastructure and are never serialized in learner state or evidence units.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.srq_fly.learner import SquareRootFLYLearner, _tensor_bytes
from tools import srq_fly_d0 as d0
from tools.srq_fly_d0 import _expand_cross, _solve, _targets
from tools.srq_fly_d3_cub import _verify_projection_prefix
from tools.tail_fly_phasea import _git_provenance, _load_unit, _save_unit, _unit_path
from tools.twa_fly_pilot import (
    _dense_codes,
    _prepare_code_cache,
    _sequence_sha256,
    _sha256_bytes,
    _sha256_file,
    _tensor_content_sha256,
)
from tools.experiment_runner import split


MANIFEST_TOP_KEYS = {
    "schema_version", "study_id", "single_use_protocol", "seeds", "backbone",
    "method_identity", "representation", "datasets", "reporting",
}
DATASET_KEYS = {"cifar100", "cub200", "imagenetr"}
METHODS = (
    "exact_fly_10000", "srq_fly_10000",
    "exact_fly_state_matched", "raw_ridge",
)
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _environment(device_name: str | None = None) -> dict:
    def package(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    device = None if device_name is None else torch.device(device_name)
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": package("torchvision"),
        "timm": package("timm"), "cuda": torch.version.cuda,
        "cudnn": None if not torch.backends.cudnn.is_available() else torch.backends.cudnn.version(),
        "requested_device": device_name,
        "gpu": (
            torch.cuda.get_device_name(device)
            if device is not None and device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
    }


def _read_manifest(path: Path) -> dict:
    manifest = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if set(manifest) != MANIFEST_TOP_KEYS or manifest.get("schema_version") != 1:
        raise ValueError("held-out manifest top-level schema mismatch")
    if manifest.get("single_use_protocol") is not True:
        raise ValueError("held-out manifest must declare single-use protocol")
    if manifest["seeds"] != [2025, 2026, 2027, 2028, 2029, 2030]:
        raise ValueError("held-out seeds differ from the locked six-seed protocol")
    if set(manifest["datasets"]) != DATASET_KEYS:
        raise ValueError("held-out manifest must contain exactly three datasets")
    backbone = manifest["backbone"]
    if (
        backbone.get("feature_dim") != 768
        or backbone.get("model_name") != "vit_base_patch16_224"
        or backbone.get("checkpoint_sha256")
        != "32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b"
        or backbone.get("preprocessing") != "vit"
    ):
        raise ValueError("backbone contract mismatch")
    representation = manifest["representation"]
    if (
        representation.get("large_expand_dim") != 10000
        or representation.get("synaptic_degree") != 300
        or representation.get("coding_level") != 0.3
        or representation.get("block_size") != 256
        or representation.get("group_size") != 64
        or representation.get("statistics_dtype") != "float32"
        or representation.get("solver_dtype") != "float32"
        or representation.get("raw_statistics_dtype") != "float64"
    ):
        raise ValueError("representation contract mismatch")
    expected_dimensions = {"cifar100": (100, 10, 4409), "cub200": (200, 20, 4518), "imagenetr": (200, 20, 4518)}
    for key, (classes, tasks, matched) in expected_dimensions.items():
        dataset = manifest["datasets"][key]
        if (
            dataset.get("num_classes") != classes
            or dataset.get("num_tasks") != tasks
            or dataset.get("matched_expand_dim") != matched
            or min(dataset.get("train_samples", 0), dataset.get("test_samples", 0)) <= 0
            or min(dataset.get("fly_ridge_lambda", 0), dataset.get("raw_ridge_lambda", 0)) <= 0
        ):
            raise ValueError(f"invalid locked dataset contract: {key}")
    reporting = manifest["reporting"]
    if (
        reporting.get("test_tuning_allowed") is not False
        or reporting.get("accuracy_based_early_stop") is not False
        or reporting.get("primary_methods") != list(METHODS)
        or reporting.get("maximum_solver_relative_residual", 0) <= 0
    ):
        raise ValueError("reporting contract mismatch")
    return manifest


def _verify_method_identity(manifest: dict) -> dict:
    identity = manifest["method_identity"]
    observed = {
        "learner_sha256": _sha256_file(ROOT / "methods/srq_fly/learner.py"),
        "storage_sha256": _sha256_file(ROOT / "methods/srq_fly/storage.py"),
        "flyhash_sha256": _sha256_file(ROOT / "models/flyhash.py"),
    }
    if observed != identity:
        raise ValueError(f"SRQ/FLY method source identity mismatch: {observed}")
    return observed


def _zip_json(path: Path, member: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name == member or name.endswith("/" + member)]
        if len(matches) != 1:
            raise ValueError(f"expected one {member} in {path.name}; found {matches}")
        return json.loads(archive.read(matches[0]), object_pairs_hook=_reject_duplicate_keys)


def _verify_one_evidence(directory: Path, evidence: dict) -> tuple[Path, dict]:
    path = directory / evidence["artifact_name"]
    if not path.is_file():
        raise FileNotFoundError(f"missing train-only evidence: {path}")
    if path.stat().st_size != evidence["artifact_size"]:
        raise ValueError(f"train-only artifact size mismatch: {path.name}")
    if _sha256_file(path) != evidence["artifact_sha256"]:
        raise ValueError(f"train-only artifact SHA-256 mismatch: {path.name}")
    payload = _zip_json(path, evidence["result_member"])
    if (
        payload.get("status") != evidence["status"]
        or payload.get("uses_test_set") is not False
        or payload.get("held_out_test_authorized") is not False
    ):
        raise ValueError(f"train-only result contract mismatch: {path.name}")
    return path, payload


def verify_train_only_evidence(manifest: dict, artifact_dir: Path) -> list[dict]:
    verified = []
    for dataset_key, dataset in manifest["datasets"].items():
        evidence = dataset["train_only_evidence"]
        path, payload = _verify_one_evidence(artifact_dir, evidence)
        if dataset_key == "cifar100":
            fly_lambda = payload.get("selected_fly_and_srq_lambda")
            raw_lambda = payload.get("fixed_raw_ridge_lambda")
        elif dataset_key == "cub200":
            fly_lambda = payload.get("fixed_fly_ridge_lambda")
            raw_lambda = payload.get("selected_raw_ridge_lambda")
        else:
            fly_lambda = payload.get("provenance", {}).get("selected_lambda")
            raw_lambda = None
        if float(fly_lambda) != float(dataset["fly_ridge_lambda"]):
            raise ValueError(f"locked FLY lambda mismatch for {dataset_key}")
        if raw_lambda is not None and float(raw_lambda) != float(dataset["raw_ridge_lambda"]):
            raise ValueError(f"locked raw-Ridge lambda mismatch for {dataset_key}")
        verified.append({
            "dataset_key": dataset_key, "artifact_name": path.name,
            "artifact_sha256": evidence["artifact_sha256"],
            "status": payload["status"], "uses_test_set": False,
            "fly_ridge_lambda": float(fly_lambda),
            "raw_ridge_lambda": None if raw_lambda is None else float(raw_lambda),
        })
        raw_evidence = dataset.get("raw_ridge_train_only_evidence")
        if raw_evidence is not None:
            raw_path, raw_payload = _verify_one_evidence(artifact_dir, raw_evidence)
            raw_results = [
                item for item in raw_payload.get("results", [])
                if item.get("method") == "raw_ridge"
            ]
            if len(raw_results) != 1 or float(raw_results[0].get("ridge_lambda", -1)) != float(dataset["raw_ridge_lambda"]):
                raise ValueError(f"raw-Ridge evidence mismatch for {dataset_key}")
            verified.append({
                "dataset_key": dataset_key, "artifact_name": raw_path.name,
                "artifact_sha256": raw_evidence["artifact_sha256"],
                "status": raw_payload["status"], "uses_test_set": False,
                "raw_ridge_lambda": float(raw_results[0]["ridge_lambda"]),
            })
    return verified


def authorize(manifest_path: Path, artifact_dir: Path, output_root: Path, require_clean_git: bool) -> dict:
    manifest = _read_manifest(manifest_path)
    identities = _verify_method_identity(manifest)
    evidence = verify_train_only_evidence(manifest, artifact_dir)
    git = _git_provenance()
    if require_clean_git and git["git_dirty"] is not False:
        raise RuntimeError("held-out authorization requires a clean Git worktree")
    record = {
        "schema_version": 1,
        "study_id": manifest["study_id"],
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": _sha256_file(manifest_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "git_commit": git["git_commit"], "git_dirty": git["git_dirty"],
        "method_identity": identities, "train_only_evidence": evidence,
        "environment": _environment(),
        "test_tuning_allowed": False, "accuracy_based_early_stop": False,
    }
    record["authorization_id"] = _sha256_bytes(
        json.dumps(record, sort_keys=True).encode("utf-8")
    )
    path = output_root / "heldout_authorization.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        immutable = ("study_id", "manifest_sha256", "runner_sha256", "git_commit", "method_identity", "train_only_evidence")
        if any(previous.get(key) != record.get(key) for key in immutable):
            raise RuntimeError("existing held-out authorization belongs to a different immutable context")
        print(f"AUTHORIZATION restored id={previous['authorization_id']}", flush=True)
        return previous
    _atomic_json(path, record)
    print(f"AUTHORIZATION LOCKED id={record['authorization_id']}", flush=True)
    return record


def _validate_authorization(path: Path, manifest_path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError("held-out authorization record is missing")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("manifest_sha256") != _sha256_file(manifest_path)
        or record.get("runner_sha256") != _sha256_file(Path(__file__).resolve())
        or record.get("test_tuning_allowed") is not False
        or record.get("accuracy_based_early_stop") is not False
    ):
        raise ValueError("held-out authorization identity mismatch")
    return record


def _validate_dataset_audit(path: Path | None, dataset_key: str, config: dict) -> dict | None:
    if dataset_key == "cifar100":
        if path is not None:
            raise ValueError("CIFAR does not use an ImageFolder identity audit")
        return None
    if path is None or not path.is_file():
        raise FileNotFoundError(f"dataset audit required for {dataset_key}")
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset_identity_sha256": config["dataset_identity_sha256"],
        "class_mapping_sha256": config["class_mapping_sha256"],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"dataset audit mismatch for {key}")
    if report.get("train", {}).get("content_manifest_sha256") != config["train_content_manifest_sha256"]:
        raise ValueError("train content-manifest mismatch")
    if report.get("test", {}).get("content_manifest_sha256") != config["test_content_manifest_sha256"]:
        raise ValueError("test content-manifest mismatch")
    if dataset_key == "cub200" and report.get("cross_split_duplicate_content_count") != 0:
        raise ValueError("CUB train/test overlap is forbidden")
    if dataset_key == "imagenetr" and (
        report.get("cross_split_duplicate_content_count") != config["cross_split_duplicate_content_count"]
        or report.get("cross_split_conflicting_label_duplicate_count") != config["cross_split_conflicting_label_duplicate_count"]
    ):
        raise ValueError("ImageNet-R legacy overlap disclosure mismatch")
    return report


def _validate_feature_cache(cache_dir: Path, manifest: dict, dataset_key: str) -> tuple[dict, dict, dict]:
    config = manifest["datasets"][dataset_key]
    metadata_path, train_path, test_path = (
        cache_dir / "metadata.json", cache_dir / "train.pt", cache_dir / "test.pt"
    )
    if not all(path.is_file() for path in (metadata_path, train_path, test_path)):
        raise FileNotFoundError("held-out feature cache requires metadata.json, train.pt and test.pt")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    backbone = manifest["backbone"]
    if (
        metadata.get("dataset") != config["dataset"]
        or metadata.get("backbone_model") != backbone["model_name"]
        or metadata.get("checkpoint_sha256") != backbone["checkpoint_sha256"]
        or metadata.get("preprocessing") != backbone["preprocessing"]
    ):
        raise ValueError("held-out feature-cache metadata mismatch")
    train = torch.load(train_path, weights_only=True, map_location="cpu")
    test = torch.load(test_path, weights_only=True, map_location="cpu")
    for name, packed, samples in (
        ("train", train, config["train_samples"]), ("test", test, config["test_samples"]),
    ):
        if (
            set(packed) != {"features", "labels"}
            or tuple(packed["features"].shape) != (samples, backbone["feature_dim"])
            or tuple(packed["labels"].shape) != (samples,)
            or not bool(torch.isfinite(packed["features"]).all())
            or sorted(map(int, torch.unique(packed["labels"]).tolist())) != list(range(config["num_classes"]))
        ):
            raise ValueError(f"invalid {name} feature cache")
    return train, test, metadata


def _representation_config(manifest: dict, dataset: dict, seed: int, expand_dim: int) -> dict:
    representation = manifest["representation"]
    return {
        "seed": seed, "num_classes": dataset["num_classes"],
        "representation": {
            "expand_dim": expand_dim,
            "synaptic_degree": representation["synaptic_degree"],
            "coding_level": representation["coding_level"],
            "encode_batch_size": representation["encode_batch_size"],
            "evaluation_batch_size": representation["evaluation_batch_size"],
        },
        "statistics_dtype": representation["statistics_dtype"],
        "raw_ridge_lambda": dataset["raw_ridge_lambda"],
        "solver_tolerance": manifest["reporting"]["maximum_solver_relative_residual"],
        "solver_max_iterations": 100,
    }


def _inventory(tensors: dict[str, torch.Tensor]) -> list[dict]:
    return [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "layout": str(tensor.layout), "bytes": _tensor_bytes(tensor)}
        for name, tensor in sorted(tensors.items())
    ]


def _assert_sample_free_inventory(
    inventory: list[dict], historical_rows: int, structural_dimensions: set[int]
) -> None:
    forbidden = ("sample", "history", "feature", "label", "code_cache", "indices")
    for item in inventory:
        if any(token in item["name"].lower() for token in forbidden):
            raise AssertionError(f"forbidden persistent tensor: {item['name']}")
        if (
            historical_rows > 0
            and historical_rows not in structural_dimensions
            and historical_rows in item["shape"]
        ):
            raise AssertionError(f"historical sample dimension in {item['name']}")


def _runtime_state_contract(manifest: dict, dataset: dict, large_projection: torch.Tensor, matched_projection: torch.Tensor) -> dict:
    degree = manifest["representation"]["synaptic_degree"]
    large_nominal = manifest["representation"]["large_expand_dim"] * degree
    matched_nominal = dataset["matched_expand_dim"] * degree
    large_missing = large_nominal - int(large_projection.values().numel())
    matched_missing = matched_nominal - int(matched_projection.values().numel())
    maximum = dataset["maximum_missing_projection_entries"]
    if min(large_missing, matched_missing) < 0 or max(large_missing, matched_missing) > maximum:
        raise ValueError("realized sparse projection is outside the locked stored-entry bounds")
    classes = dataset["num_classes"]
    raw_dim = manifest["backbone"]["feature_dim"]
    return {
        "large_missing_projection_entries": large_missing,
        "matched_missing_projection_entries": matched_missing,
        "exact_large_bytes": dataset["nominal_exact_large_bytes"] - 12 * large_missing,
        "srq_large_bytes": dataset["nominal_srq_large_bytes"] - 12 * large_missing,
        "exact_matched_bytes": dataset["nominal_exact_matched_bytes"] - 12 * matched_missing,
        "raw_ridge_bytes": (raw_dim * raw_dim + 2 * raw_dim * classes + classes) * 8,
    }


def _task_predictions(
    *, weights: torch.Tensor, class_ids: list[int], parts: list[torch.Tensor], task: int,
    code_indices: torch.Tensor, code_values: torch.Tensor, labels: torch.Tensor,
    dimension: int, batch_size: int,
) -> tuple[list[float], list[torch.Tensor], list[torch.Tensor]]:
    accuracies, predictions, logits_cpu = [], [], []
    for previous in range(task + 1):
        indices = parts[previous]
        task_predictions, task_logits = [], []
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            codes = _dense_codes(code_indices[selected], code_values[selected], dimension, weights.device, weights.dtype)
            logits = codes @ weights
            columns = logits.argmax(1).detach().cpu().tolist()
            task_predictions.append(torch.tensor([class_ids[column] for column in columns]))
            task_logits.append(logits.detach().cpu().to(torch.float64))
        predicted = torch.cat(task_predictions)
        truth = labels[indices].cpu()
        accuracies.append(100.0 * float((predicted == truth).float().mean()))
        predictions.append(predicted)
        logits_cpu.append(torch.cat(task_logits))
    return accuracies, predictions, logits_cpu


def _result_metrics(matrix: list[list[float]]) -> dict:
    stage_accuracy = [statistics.fmean(row) for row in matrix]
    final_row = matrix[-1]
    forgetting = 0.0
    if len(matrix) > 1:
        drops = []
        for task in range(len(matrix) - 1):
            past = [matrix[stage][task] for stage in range(task, len(matrix))]
            drops.append(max(past) - final_row[task])
        forgetting = statistics.fmean(drops)
    return {
        "stage_accuracy": stage_accuracy,
        "final_accuracy": statistics.fmean(final_row),
        "average_incremental_accuracy": statistics.fmean(stage_accuracy),
        "forgetting": forgetting,
    }


def _evaluate_paired(
    *, manifest: dict, dataset: dict, seed: int, stream: dict,
    code_indices: torch.Tensor, code_values: torch.Tensor, projection: torch.Tensor,
    training_parts: list[torch.Tensor], test_parts: list[torch.Tensor], device: torch.device,
) -> dict:
    common = manifest["representation"]
    dimension = common["large_expand_dim"]
    dtype = torch.float32
    gram = torch.zeros((dimension, dimension), device=device, dtype=dtype)
    cross = torch.zeros((dimension, 0), device=device, dtype=dtype)
    counts = torch.zeros(0, device=device, dtype=dtype)
    class_ids = []
    learner = SquareRootFLYLearner(
        storage_mode="int8", feature_dim=manifest["backbone"]["feature_dim"],
        expand_dim=dimension, synaptic_degree=common["synaptic_degree"],
        coding_level=common["coding_level"], ridge_lambda=dataset["fly_ridge_lambda"],
        block_size=common["block_size"], group_size=common["group_size"], seed=seed,
        device=device, statistics_dtype=dtype, solver_dtype=dtype, projection=projection,
    )
    exact_matrix, srq_matrix = [], []
    exact_residuals, srq_residuals, agreements, logit_errors = [], [], [], []
    exact_state_by_task, srq_state_by_task, timing = [], [], []
    exact_weights = None
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        code_started = time.perf_counter()
        codes = _dense_codes(code_indices[indices], code_values[indices], dimension, device, dtype)
        code_seconds = time.perf_counter() - code_started
        exact_update_started = time.perf_counter()
        labels = stream["labels"][indices]
        new_ids = sorted(set(class_ids) | set(map(int, labels.tolist())))
        cross, counts = _expand_cross(cross, counts, class_ids, new_ids)
        class_ids = new_ids
        targets = _targets(labels, class_ids, device=device, dtype=dtype)
        gram += codes.T @ codes
        cross += codes.T @ targets
        counts += targets.sum(0)
        system = gram + dataset["fly_ridge_lambda"] * torch.eye(dimension, device=device, dtype=dtype)
        exact_weights, exact_residual = _solve(system, cross)
        exact_update_seconds = time.perf_counter() - exact_update_started
        srq_update_started = time.perf_counter()
        learner.update_codes(codes, labels)
        learner.assert_exemplar_free_state()
        srq_update_seconds = time.perf_counter() - srq_update_started
        del codes, system
        exact_inference_started = time.perf_counter()
        exact_row, exact_predictions, exact_logits = _task_predictions(
            weights=exact_weights, class_ids=class_ids, parts=test_parts, task=task,
            code_indices=code_indices, code_values=code_values, labels=stream["labels"],
            dimension=dimension, batch_size=common["evaluation_batch_size"],
        )
        exact_inference_seconds = time.perf_counter() - exact_inference_started
        srq_inference_started = time.perf_counter()
        srq_row, srq_predictions, srq_logits = _task_predictions(
            weights=learner.weights, class_ids=class_ids, parts=test_parts, task=task,
            code_indices=code_indices, code_values=code_values, labels=stream["labels"],
            dimension=dimension, batch_size=common["evaluation_batch_size"],
        )
        srq_inference_seconds = time.perf_counter() - srq_inference_started
        agreements.append(statistics.fmean([
            float((left == right).float().mean()) for left, right in zip(exact_predictions, srq_predictions)
        ]))
        numerator = sum(float(((right - left) ** 2).sum()) for left, right in zip(exact_logits, srq_logits))
        denominator = max(sum(float((left ** 2).sum()) for left in exact_logits), 1.0)
        logit_errors.append(math.sqrt(numerator / denominator))
        exact_matrix.append(exact_row); srq_matrix.append(srq_row)
        exact_residuals.append(exact_residual)
        srq_residuals.append(float(learner.diagnostics["solver_relative_residual"]))
        exact_tensors = {"projection": projection, "G": gram, "Q": cross, "counts": counts, "weights": exact_weights}
        exact_inventory = _inventory(exact_tensors)
        srq_inventory = _inventory(learner.persistent_tensors())
        structural = {dimension, manifest["backbone"]["feature_dim"], len(class_ids)}
        _assert_sample_free_inventory(exact_inventory, learner.total_rows, structural)
        _assert_sample_free_inventory(srq_inventory, learner.total_rows, structural)
        exact_state_by_task.append(sum(item["bytes"] for item in exact_inventory))
        srq_state_by_task.append(sum(item["bytes"] for item in srq_inventory))
        timing.append({
            "task": task + 1, "shared_code_materialization_seconds": code_seconds,
            "exact_update_seconds": exact_update_seconds,
            "srq_update_seconds": srq_update_seconds,
            "exact_inference_seconds": exact_inference_seconds,
            "srq_inference_seconds": srq_inference_seconds,
        })
        print(
            f"TASK paired seed={seed} {task+1}/{len(training_parts)} "
            f"exact={statistics.fmean(exact_row):.4f} srq={statistics.fmean(srq_row):.4f} "
            f"agree={100*agreements[-1]:.3f}%",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    exact_metrics, srq_metrics = _result_metrics(exact_matrix), _result_metrics(srq_matrix)
    exact_inventory = _inventory({"projection": projection, "G": gram, "Q": cross, "counts": counts, "weights": exact_weights})
    srq_inventory = _inventory(learner.persistent_tensors())
    base = {"status": "complete", "uses_test_set": True, "exemplar_free": True, "ridge_lambda": dataset["fly_ridge_lambda"]}
    return {
        "status": "complete", "uses_test_set": True,
        "exact": {**base, "method": "exact_fly_10000", **exact_metrics,
                  "accuracy_matrix": exact_matrix, "persistent_state_bytes": exact_state_by_task[-1],
                  "persistent_state_bytes_by_task": exact_state_by_task,
                  "persistent_tensor_inventory": exact_inventory,
                  "maximum_solver_relative_residual": max(exact_residuals),
                  "total_update_seconds": sum(item["exact_update_seconds"] for item in timing),
                  "total_inference_seconds": sum(item["exact_inference_seconds"] for item in timing),
                  "timing": timing, "seconds": time.perf_counter() - started},
        "srq": {**base, "method": "srq_fly_10000", **srq_metrics,
                "accuracy_matrix": srq_matrix, "persistent_state_bytes": srq_state_by_task[-1],
                "persistent_state_bytes_by_task": srq_state_by_task,
                "persistent_tensor_inventory": srq_inventory,
                "maximum_solver_relative_residual": max(srq_residuals),
                "minimum_prediction_agreement": min(agreements),
                "maximum_relative_logit_frobenius_error": max(logit_errors),
                "total_update_seconds": sum(item["srq_update_seconds"] for item in timing),
                "total_inference_seconds": sum(item["srq_inference_seconds"] for item in timing),
                "timing": timing,
                "seconds": time.perf_counter() - started},
        "prediction_agreement_by_task": agreements,
        "relative_logit_frobenius_error_by_task": logit_errors,
        "timing": timing,
    }


def _evaluate_exact_matched(
    *, manifest: dict, dataset: dict, seed: int, stream: dict,
    code_indices: torch.Tensor, code_values: torch.Tensor, projection: torch.Tensor,
    training_parts: list[torch.Tensor], test_parts: list[torch.Tensor], device: torch.device,
) -> dict:
    common = manifest["representation"]
    dimension = dataset["matched_expand_dim"]
    dtype = torch.float32
    gram = torch.zeros((dimension, dimension), device=device, dtype=dtype)
    cross = torch.zeros((dimension, 0), device=device, dtype=dtype)
    counts = torch.zeros(0, device=device, dtype=dtype)
    class_ids, matrix, residuals, states, timing = [], [], [], [], []
    weights = None
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        update_started = time.perf_counter()
        codes = _dense_codes(code_indices[indices], code_values[indices], dimension, device, dtype)
        labels = stream["labels"][indices]
        new_ids = sorted(set(class_ids) | set(map(int, labels.tolist())))
        cross, counts = _expand_cross(cross, counts, class_ids, new_ids); class_ids = new_ids
        targets = _targets(labels, class_ids, device=device, dtype=dtype)
        gram += codes.T @ codes; cross += codes.T @ targets; counts += targets.sum(0)
        system = gram + dataset["fly_ridge_lambda"] * torch.eye(dimension, device=device, dtype=dtype)
        weights, residual = _solve(system, cross); residuals.append(residual)
        update_seconds = time.perf_counter() - update_started
        inference_started = time.perf_counter()
        row, _, _ = _task_predictions(
            weights=weights, class_ids=class_ids, parts=test_parts, task=task,
            code_indices=code_indices, code_values=code_values, labels=stream["labels"],
            dimension=dimension, batch_size=common["evaluation_batch_size"],
        )
        inference_seconds = time.perf_counter() - inference_started
        matrix.append(row)
        inventory = _inventory({"projection": projection, "G": gram, "Q": cross, "counts": counts, "weights": weights})
        _assert_sample_free_inventory(
            inventory, sum(len(part) for part in training_parts[:task + 1]),
            {dimension, manifest["backbone"]["feature_dim"], len(class_ids)},
        )
        states.append(sum(item["bytes"] for item in inventory))
        timing.append({"task": task + 1, "update_seconds": update_seconds, "inference_seconds": inference_seconds})
        print(f"TASK matched seed={seed} {task+1}/{len(training_parts)} AA={statistics.fmean(row):.4f}", flush=True)
        del codes, system
    return {
        "method": "exact_fly_state_matched", "status": "complete", "uses_test_set": True,
        "exemplar_free": True, "ridge_lambda": dataset["fly_ridge_lambda"],
        **_result_metrics(matrix), "accuracy_matrix": matrix,
        "persistent_state_bytes": states[-1], "persistent_state_bytes_by_task": states,
        "persistent_tensor_inventory": inventory,
        "maximum_solver_relative_residual": max(residuals), "timing": timing,
        "total_update_seconds": sum(item["update_seconds"] for item in timing),
        "total_inference_seconds": sum(item["inference_seconds"] for item in timing),
        "seconds": time.perf_counter() - started,
    }


def _evaluate_raw(
    *, manifest: dict, dataset: dict, seed: int, stream: dict,
    training_parts: list[torch.Tensor], test_parts: list[torch.Tensor], device: torch.device,
) -> dict:
    dimension = manifest["backbone"]["feature_dim"]
    dtype = torch.float64
    gram = torch.zeros((dimension, dimension), device=device, dtype=dtype)
    cross = torch.zeros((dimension, 0), device=device, dtype=dtype)
    counts = torch.zeros(0, device=device, dtype=dtype)
    class_ids, matrix, residuals, states, timing = [], [], [], [], []
    weights = None
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        update_started = time.perf_counter()
        values = stream["features"][indices].to(device=device, dtype=dtype)
        labels = stream["labels"][indices]
        new_ids = sorted(set(class_ids) | set(map(int, labels.tolist())))
        cross, counts = _expand_cross(cross, counts, class_ids, new_ids); class_ids = new_ids
        targets = _targets(labels, class_ids, device=device, dtype=dtype)
        gram += values.T @ values; cross += values.T @ targets; counts += targets.sum(0)
        system = gram + dataset["raw_ridge_lambda"] * torch.eye(dimension, device=device, dtype=dtype)
        weights, residual = _solve(system, cross); residuals.append(residual)
        update_seconds = time.perf_counter() - update_started
        inference_started = time.perf_counter()
        row = []
        for previous in range(task + 1):
            selected = test_parts[previous]
            correct = rows = 0
            for start in range(0, len(selected), 256):
                batch = selected[start:start + 256]
                logits = stream["features"][batch].to(device=device, dtype=dtype) @ weights
                columns = logits.argmax(1).cpu().tolist()
                predictions = torch.tensor([class_ids[column] for column in columns])
                correct += int((predictions == stream["labels"][batch]).sum()); rows += len(batch)
            row.append(100.0 * correct / max(rows, 1))
        inference_seconds = time.perf_counter() - inference_started
        matrix.append(row)
        inventory = _inventory({"G": gram, "Q": cross, "counts": counts, "weights": weights})
        _assert_sample_free_inventory(
            inventory, sum(len(part) for part in training_parts[:task + 1]),
            {dimension, len(class_ids)},
        )
        states.append(sum(item["bytes"] for item in inventory))
        timing.append({"task": task + 1, "update_seconds": update_seconds, "inference_seconds": inference_seconds})
        print(f"TASK raw seed={seed} {task+1}/{len(training_parts)} AA={statistics.fmean(row):.4f}", flush=True)
    return {
        "method": "raw_ridge", "status": "complete", "uses_test_set": True,
        "exemplar_free": True, "ridge_lambda": dataset["raw_ridge_lambda"],
        **_result_metrics(matrix), "accuracy_matrix": matrix,
        "persistent_state_bytes": states[-1], "persistent_state_bytes_by_task": states,
        "persistent_tensor_inventory": inventory,
        "maximum_solver_relative_residual": max(residuals), "timing": timing,
        "total_update_seconds": sum(item["update_seconds"] for item in timing),
        "total_inference_seconds": sum(item["inference_seconds"] for item in timing),
        "seconds": time.perf_counter() - started,
    }


def _known_numerical_failure(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, torch.linalg.LinAlgError) or any(token in text for token in (
        "cholesky", "positive definite", "positive-definite", "solver",
    ))


def _execute_unit(path: Path, context_sha: str, label: str, evaluator) -> dict:
    result = _load_unit(path, context_sha)
    if result is not None:
        print(f"UNIT RESTORED {label}", flush=True)
        return result
    print(f"UNIT START {label}", flush=True)
    try:
        result = evaluator()
    except (RuntimeError, torch.linalg.LinAlgError) as error:
        if not _known_numerical_failure(error):
            raise
        result = {
            "status": "numerical_failure", "failure": f"{type(error).__name__}: {error}",
            "uses_test_set": True, "exemplar_free": True,
        }
    result = _save_unit(path, context_sha, result)
    print(f"UNIT DONE {label} status={result['status']}", flush=True)
    return result


def evaluate_dataset(
    *, manifest_path: Path, dataset_key: str, feature_cache_dir: Path,
    code_cache_root: Path, output_root: Path, authorization_path: Path,
    dataset_audit_path: Path | None, device_name: str,
) -> dict:
    manifest = _read_manifest(manifest_path)
    _verify_method_identity(manifest)
    authorization = _validate_authorization(authorization_path, manifest_path)
    if dataset_key not in manifest["datasets"]:
        raise ValueError(f"unknown dataset key: {dataset_key}")
    dataset = manifest["datasets"][dataset_key]
    audit = _validate_dataset_audit(dataset_audit_path, dataset_key, dataset)
    train, test, metadata = _validate_feature_cache(feature_cache_dir, manifest, dataset_key)
    train_sha, test_sha = _sha256_file(feature_cache_dir / "train.pt"), _sha256_file(feature_cache_dir / "test.pt")
    stream = {
        "features": torch.cat((train["features"], test["features"])),
        "labels": torch.cat((train["labels"], test["labels"])),
    }
    source_sha = _sha256_bytes((train_sha + test_sha).encode("ascii"))
    offset = len(train["labels"])
    device = torch.device(device_name)
    dataset_output = output_root / dataset_key
    dataset_output.mkdir(parents=True, exist_ok=True)
    seed_results = []
    for seed in manifest["seeds"]:
        print(f"SEED START dataset={dataset_key} seed={seed}", flush=True)
        class_order = random.Random(seed).sample(range(dataset["num_classes"]), dataset["num_classes"])
        training_parts = split(train["labels"], class_order, dataset["num_tasks"])
        test_parts = [part + offset for part in split(test["labels"], class_order, dataset["num_tasks"])]
        seed_cache = code_cache_root / dataset_key / f"seed_{seed}"
        large_config = _representation_config(manifest, dataset, seed, manifest["representation"]["large_expand_dim"])
        matched_config = _representation_config(manifest, dataset, seed, dataset["matched_expand_dim"])
        large = _prepare_code_cache(train=stream, train_sha256=source_sha, cache_dir=seed_cache / "large", config=large_config, device=device_name)
        matched = _prepare_code_cache(train=stream, train_sha256=source_sha, cache_dir=seed_cache / "matched", config=matched_config, device=device_name)
        prefix = _verify_projection_prefix(large[3], matched[3])
        runtime_state = _runtime_state_contract(manifest, dataset, large[3], matched[3])
        context = {
            "manifest_sha256": _sha256_file(manifest_path), "authorization_id": authorization["authorization_id"],
            "dataset_key": dataset_key, "seed": seed, "train_sha256": train_sha, "test_sha256": test_sha,
            "class_order": class_order,
            "training_indices_sha256": _sequence_sha256(training_parts),
            "test_indices_sha256": _sequence_sha256(test_parts),
            "large_code_identity": large[2]["identity_sha256"], "matched_code_identity": matched[2]["identity_sha256"],
            "large_projection_sha256": _tensor_content_sha256(large[3]), "matched_projection_sha256": _tensor_content_sha256(matched[3]),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
        }
        context_sha = _sha256_bytes(json.dumps(context, sort_keys=True).encode())
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        paired = _execute_unit(
            _unit_path(dataset_output, f"seed_{seed}_paired"), context_sha, f"{dataset_key}/{seed}/paired",
            lambda: _evaluate_paired(
                manifest=manifest, dataset=dataset, seed=seed, stream=stream,
                code_indices=large[0], code_values=large[1], projection=large[3],
                training_parts=training_parts, test_parts=test_parts, device=device,
            ),
        )
        matched_result = _execute_unit(
            _unit_path(dataset_output, f"seed_{seed}_matched"), context_sha, f"{dataset_key}/{seed}/matched",
            lambda: _evaluate_exact_matched(
                manifest=manifest, dataset=dataset, seed=seed, stream=stream,
                code_indices=matched[0], code_values=matched[1], projection=matched[3],
                training_parts=training_parts, test_parts=test_parts, device=device,
            ),
        )
        raw = _execute_unit(
            _unit_path(dataset_output, f"seed_{seed}_raw"), context_sha, f"{dataset_key}/{seed}/raw",
            lambda: _evaluate_raw(
                manifest=manifest, dataset=dataset, seed=seed, stream=stream,
                training_parts=training_parts, test_parts=test_parts, device=device,
            ),
        )
        peak = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else None
        methods = {
            "exact_fly_10000": paired.get("exact") if paired.get("status") == "complete" else paired,
            "srq_fly_10000": paired.get("srq") if paired.get("status") == "complete" else paired,
            "exact_fly_state_matched": matched_result,
            "raw_ridge": raw,
        }
        expected_state = {
            "exact_fly_10000": runtime_state["exact_large_bytes"],
            "srq_fly_10000": runtime_state["srq_large_bytes"],
            "exact_fly_state_matched": runtime_state["exact_matched_bytes"],
            "raw_ridge": runtime_state["raw_ridge_bytes"],
        }
        for name, result in methods.items():
            if result.get("status") == "complete" and result.get("persistent_state_bytes") != expected_state[name]:
                raise AssertionError(
                    f"persistent state mismatch for {dataset_key}/{seed}/{name}: "
                    f"{result.get('persistent_state_bytes')} != {expected_state[name]}"
                )
        seed_results.append({
            "seed": seed, "class_order": class_order, "context": context,
            "projection_prefix": prefix, "methods": methods,
            "runtime_state_contract": runtime_state,
            "paired_diagnostics": {
                "prediction_agreement_by_task": paired.get("prediction_agreement_by_task"),
                "relative_logit_frobenius_error_by_task": paired.get("relative_logit_frobenius_error_by_task"),
            },
            "peak_runtime_gpu_bytes": peak,
        })
        print(f"SEED DONE dataset={dataset_key} seed={seed}", flush=True)
    tolerance = manifest["reporting"]["maximum_solver_relative_residual"]
    failures = [
        {"seed": item["seed"], "method": name, "status": result.get("status")}
        for item in seed_results for name, result in item["methods"].items()
        if result.get("status") != "complete"
    ]
    failures.extend(
        {"seed": item["seed"], "method": name, "status": "numerical_tolerance_failed"}
        for item in seed_results for name, result in item["methods"].items()
        if result.get("status") == "complete"
        and float(result.get("maximum_solver_relative_residual", float("inf"))) > tolerance
    )
    payload = {
        "schema_version": 1, "study_id": manifest["study_id"], "dataset_key": dataset_key,
        "dataset": dataset["dataset"], "status": "COMPLETE" if not failures else "COMPLETE_WITH_METHOD_FAILURES",
        "uses_test_set": True, "test_tuning_allowed": False, "accuracy_based_early_stop": False,
        "authorization_id": authorization["authorization_id"],
        "exact_command": [sys.executable, *sys.argv],
        "environment": _environment(device_name),
        "provenance": {"manifest_sha256": _sha256_file(manifest_path), "train_sha256": train_sha, "test_sha256": test_sha},
        "source_feature_metadata": metadata, "dataset_audit": audit,
        "legacy_processed_split_disclosure": (
            "19 cross-split duplicate hashes, including 18 conflicting-label hashes; not content-disjoint"
            if dataset_key == "imagenetr" else None
        ),
        "feature_cache_disk_bytes": sum(path.stat().st_size for path in feature_cache_dir.iterdir() if path.is_file()),
        "wta_cache_disk_bytes": sum(path.stat().st_size for path in (code_cache_root / dataset_key).rglob("*") if path.is_file()),
        "seed_results": seed_results, "failures": failures,
    }
    _atomic_json(dataset_output / "heldout_results.json", payload)
    print(f"DATASET COMPLETE {dataset_key} status={payload['status']}", flush=True)
    return payload


def _mean_std_ci(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "sample_std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = statistics.stdev(values)
    critical = T_CRITICAL_95.get(len(values) - 1, 1.96)
    half = critical * std / math.sqrt(len(values))
    return {"n": len(values), "mean": mean, "sample_std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def summarize(manifest_path: Path, output_root: Path) -> dict:
    manifest = _read_manifest(manifest_path)
    tables, rows, paired = {}, [], {}
    for dataset_key in manifest["datasets"]:
        path = output_root / dataset_key / "heldout_results.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing held-out result for {dataset_key}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("uses_test_set") is not True or payload.get("test_tuning_allowed") is not False:
            raise ValueError(f"invalid held-out result contract for {dataset_key}")
        tables[dataset_key] = {}
        for method in METHODS:
            results = [item["methods"][method] for item in payload["seed_results"]]
            if any(item.get("status") != "complete" for item in results):
                tables[dataset_key][method] = {"status": "incomplete"}
                rows.append({"dataset": dataset_key, "method": method, "status": "incomplete"})
                continue
            metric_names = (
                "final_accuracy", "average_incremental_accuracy", "forgetting",
                "persistent_state_bytes", "total_update_seconds", "total_inference_seconds",
            )
            summary = {
                metric: _mean_std_ci([float(item[metric]) for item in results])
                for metric in metric_names
            }
            tables[dataset_key][method] = {"status": "complete", **summary}
            rows.append({
                "dataset": dataset_key, "method": method, "status": "complete",
                **{f"{metric}_{field}": values[field] for metric, values in summary.items() for field in ("mean", "sample_std", "ci95_low", "ci95_high")},
            })
        gains = []
        for item in payload["seed_results"]:
            srq, matched = item["methods"]["srq_fly_10000"], item["methods"]["exact_fly_state_matched"]
            if srq.get("status") == matched.get("status") == "complete":
                gains.append(srq["average_incremental_accuracy"] - matched["average_incremental_accuracy"])
        paired[dataset_key] = _mean_std_ci(gains) if gains else {"n": 0}
    summary = {
        "schema_version": 1, "study_id": manifest["study_id"], "status": "REPORTED_WITHOUT_ACCURACY_GATE",
        "uses_test_set": True, "manifest_sha256": _sha256_file(manifest_path),
        "dataset_method_summaries": tables, "paired_srq_minus_state_matched_fly": paired,
        "imagenetr_disclosure": "legacy processed split with 19 cross-split duplicate hashes; not content-disjoint",
    }
    _atomic_json(output_root / "three_dataset_summary.json", summary)
    with (output_root / "three_dataset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-selection")
    verify.add_argument("--manifest", required=True); verify.add_argument("--artifact-dir", required=True)
    auth = subparsers.add_parser("authorize")
    auth.add_argument("--manifest", required=True); auth.add_argument("--artifact-dir", required=True); auth.add_argument("--output-root", required=True)
    auth.add_argument("--require-clean-git", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True); evaluate.add_argument("--dataset-key", choices=sorted(DATASET_KEYS), required=True)
    evaluate.add_argument("--feature-cache-dir", required=True); evaluate.add_argument("--code-cache-root", required=True)
    evaluate.add_argument("--output-root", required=True); evaluate.add_argument("--authorization", required=True)
    evaluate.add_argument("--dataset-audit"); evaluate.add_argument("--device", default="cpu")
    report = subparsers.add_parser("summarize")
    report.add_argument("--manifest", required=True); report.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    if args.command == "verify-selection":
        manifest = _read_manifest(manifest_path); _verify_method_identity(manifest)
        print(json.dumps(verify_train_only_evidence(manifest, Path(args.artifact_dir).resolve()), indent=2))
    elif args.command == "authorize":
        authorize(manifest_path, Path(args.artifact_dir).resolve(), Path(args.output_root).resolve(), args.require_clean_git)
    elif args.command == "evaluate":
        evaluate_dataset(
            manifest_path=manifest_path, dataset_key=args.dataset_key,
            feature_cache_dir=Path(args.feature_cache_dir).resolve(), code_cache_root=Path(args.code_cache_root).resolve(),
            output_root=Path(args.output_root).resolve(), authorization_path=Path(args.authorization).resolve(),
            dataset_audit_path=None if args.dataset_audit is None else Path(args.dataset_audit).resolve(), device_name=args.device,
        )
    else:
        summarize(manifest_path, Path(args.output_root).resolve())


if __name__ == "__main__":
    main()
