"""Self-contained train-only selection and held-out evaluation for SRQ-FLY.

The static protocol owns every grid, split and replicate seed.  Selection
refuses a visible ``test.pt``; held-out extraction/evaluation refuses to run
without a matching immutable selection lock.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbone import load_model
from tools import srq_fly_d0 as d0
from tools import srq_fly_d1 as d1
from tools.srq_fly_heldout import (
    _assert_sample_free_inventory,
    _evaluate_paired,
    _evaluate_raw,
    _inventory,
    _mean_std_ci,
    _result_metrics,
)
from tools.tail_fly_phasea import _git_provenance, _load_unit, _save_unit, _unit_path
from tools.twa_fly_pilot import (
    _prepare_code_cache,
    _sequence_sha256,
    _sha256_bytes,
    _sha256_file,
    _tensor_content_sha256,
)
from utils.data_utils import load_dataset
from utils.train_utils import feature_extract, random_initialization


DATASET_KEYS = ("cifar100", "cub200", "imagenetr")
METHODS = ("exact_fly_10000", "srq_fly_10000", "raw_ridge")


def _environment(device_name: str | None = None) -> dict:
    device = None if device_name is None else torch.device(device_name)
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": device_name,
        "gpu": torch.cuda.get_device_name(device)
        if device is not None and device.type == "cuda" and torch.cuda.is_available() else None,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _read_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or set(protocol.get("datasets", {})) != set(DATASET_KEYS):
        raise ValueError("self-contained protocol schema mismatch")
    backbone, representation = protocol["backbone"], protocol["representation"]
    if (
        backbone.get("feature_dim") != 768
        or backbone.get("model_name") != "vit_base_patch16_224"
        or backbone.get("checkpoint_sha256")
        != "32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b"
        or representation.get("expand_dim") != 10000
        or representation.get("synaptic_degree") != 300
        or representation.get("coding_level") != 0.3
    ):
        raise ValueError("backbone or representation contract mismatch")
    selection = protocol["selection"]
    grid = selection.get("ridge_grid", [])
    if (
        selection.get("split_seed") != 2025
        or selection.get("outer_validation_fraction") != 0.2
        or selection.get("inner_validation_fraction") != 0.2
        or len(grid) != 12
        or sorted(set(map(float, grid))) != list(map(float, grid))
        or any(value <= 0 for value in grid)
        or len(selection.get("development_replicates", [])) != 3
    ):
        raise ValueError("train-only selection contract mismatch")
    final = protocol["final_evaluation"]
    if (
        len(final.get("replicates", [])) != 6
        or final.get("methods") != list(METHODS)
        or final.get("test_tuning_allowed") is not False
        or final.get("accuracy_based_early_stop") is not False
    ):
        raise ValueError("final evaluation contract mismatch")
    development_pairs = {
        (item["class_order_seed"], item["projection_seed"])
        for item in selection["development_replicates"]
    }
    final_pairs = {
        (item["class_order_seed"], item["projection_seed"])
        for item in final["replicates"]
    }
    if len(development_pairs) != 3 or len(final_pairs) != 6 or development_pairs & final_pairs:
        raise ValueError("development/final replicate identities must be distinct")
    return protocol


def _verify_method_identity(protocol: dict) -> dict:
    observed = {
        "learner_sha256": _sha256_file(ROOT / "methods/srq_fly/learner.py"),
        "storage_sha256": _sha256_file(ROOT / "methods/srq_fly/storage.py"),
        "flyhash_sha256": _sha256_file(ROOT / "models/flyhash.py"),
    }
    if observed != protocol["method_identity"]:
        raise ValueError("method source identity mismatch")
    return observed


def _validate_dataset_audit(path: Path | None, key: str, dataset: dict) -> dict | None:
    if key == "cifar100":
        if path is not None:
            raise ValueError("CIFAR-100 does not use an ImageFolder audit")
        return None
    if path is None or not path.is_file():
        raise FileNotFoundError(f"dataset audit required for {key}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    for field in ("dataset_identity_sha256", "class_mapping_sha256"):
        if audit.get(field) != dataset[field]:
            raise ValueError(f"dataset audit mismatch: {field}")
    if audit.get("train", {}).get("content_manifest_sha256") != dataset["train_content_manifest_sha256"]:
        raise ValueError("train content-manifest mismatch")
    if audit.get("test", {}).get("content_manifest_sha256") != dataset["test_content_manifest_sha256"]:
        raise ValueError("test content-manifest mismatch")
    if key == "cub200" and audit.get("cross_split_duplicate_content_count") != 0:
        raise ValueError("CUB cross-split content overlap is forbidden")
    if key == "imagenetr" and (
        audit.get("cross_split_duplicate_content_count") != dataset["cross_split_duplicate_content_count"]
        or audit.get("cross_split_conflicting_label_duplicate_count")
        != dataset["cross_split_conflicting_label_duplicate_count"]
    ):
        raise ValueError("ImageNet-R legacy-overlap disclosure mismatch")
    return audit


def _validate_train_cache(cache_dir: Path, protocol: dict, key: str, *, require_test: bool) -> tuple[dict, dict | None, dict]:
    dataset, backbone = protocol["datasets"][key], protocol["backbone"]
    metadata_path, train_path, test_path = cache_dir / "metadata.json", cache_dir / "train.pt", cache_dir / "test.pt"
    if not metadata_path.is_file() or not train_path.is_file():
        raise FileNotFoundError("feature cache requires metadata.json and train.pt")
    if require_test and not test_path.is_file():
        raise FileNotFoundError("held-out evaluation requires test.pt")
    if not require_test and test_path.exists():
        raise RuntimeError("train-only selection refuses a visible test.pt")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("dataset") != dataset["dataset"]
        or metadata.get("backbone_model") != backbone["model_name"]
        or metadata.get("checkpoint_sha256") != backbone["checkpoint_sha256"]
        or metadata.get("preprocessing") != backbone["preprocessing"]
    ):
        raise ValueError("feature-cache metadata mismatch")
    train = torch.load(train_path, weights_only=True, map_location="cpu")
    test = torch.load(test_path, weights_only=True, map_location="cpu") if require_test else None
    for name, packed, rows in (("train", train, dataset["train_samples"]), ("test", test, dataset["test_samples"])):
        if packed is None:
            continue
        if (
            set(packed) != {"features", "labels"}
            or tuple(packed["features"].shape) != (rows, backbone["feature_dim"])
            or tuple(packed["labels"].shape) != (rows,)
            or not bool(torch.isfinite(packed["features"]).all())
            or sorted(map(int, torch.unique(packed["labels"]).tolist())) != list(range(dataset["num_classes"]))
        ):
            raise ValueError(f"invalid {name} feature cache")
    return train, test, metadata


def _nested_parts(labels: torch.Tensor, class_order: list[int], tasks: int, split_seed: int, outer_fraction: float, inner_fraction: float):
    """Class-stratified nested split independent of model/projection randomness."""
    per_class = {}
    for class_id in sorted(map(int, torch.unique(labels).tolist())):
        indices = torch.nonzero(labels == class_id).flatten()
        generator = torch.Generator().manual_seed(split_seed * 1000 + class_id)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_outer = max(1, int(round(len(indices) * outer_fraction)))
        development, outer_validation = indices[n_outer:], indices[:n_outer]
        n_inner = max(1, int(round(len(development) * inner_fraction)))
        inner_fit, inner_validation = development[n_inner:], development[:n_inner]
        if min(len(inner_fit), len(inner_validation), len(outer_validation)) <= 0:
            raise ValueError(f"empty nested partition for class {class_id}")
        per_class[class_id] = (inner_fit, inner_validation, development, outer_validation)
    classes_per_task = len(class_order) // tasks
    grouped = [[], [], [], []]
    for task in range(tasks):
        class_ids = class_order[task * classes_per_task:(task + 1) * classes_per_task]
        for part in range(4):
            grouped[part].append(torch.cat([per_class[class_id][part] for class_id in class_ids]))
    return tuple(grouped)


def _cache_config(protocol: dict, dataset: dict, projection_seed: int) -> dict:
    rep = protocol["representation"]
    return {
        "seed": projection_seed, "num_classes": dataset["num_classes"],
        "representation": {
            "expand_dim": rep["expand_dim"], "synaptic_degree": rep["synaptic_degree"],
            "coding_level": rep["coding_level"], "encode_batch_size": rep["encode_batch_size"],
            "evaluation_batch_size": rep["evaluation_batch_size"],
        },
        "statistics_dtype": rep["statistics_dtype"], "raw_ridge_lambda": 1.0,
        "solver_tolerance": protocol["selection"]["maximum_solver_relative_residual"],
        "solver_max_iterations": 100,
    }


def _candidate_config(protocol: dict, dataset: dict, projection_seed: int, ridge: float) -> dict:
    rep = protocol["representation"]
    return {
        "seed": projection_seed, "ridge_lambda": ridge, "raw_ridge_lambda": ridge,
        "statistics_dtype": rep["statistics_dtype"], "solver_dtype": rep["solver_dtype"],
        "large_representation": {
            "expand_dim": rep["expand_dim"], "synaptic_degree": rep["synaptic_degree"],
            "coding_level": rep["coding_level"], "evaluation_batch_size": rep["evaluation_batch_size"],
        },
        "storage": {"block_size": rep["block_size"], "group_size": rep["group_size"]},
    }


def _run_unit(path: Path, context_sha: str, label: str, evaluator) -> dict:
    restored = _load_unit(path, context_sha)
    if restored is not None:
        print(f"RESTORED {label}", flush=True)
        return restored
    print(f"START {label}", flush=True)
    started = time.perf_counter()
    try:
        result = evaluator()
    except (RuntimeError, torch.linalg.LinAlgError) as error:
        result = {
            "status": "numerical_failure", "failure": f"{type(error).__name__}: {error}",
            "uses_test_set": False,
        }
    result["unit_seconds"] = time.perf_counter() - started
    result = _save_unit(path, context_sha, result)
    print(f"DONE {label} status={result['status']}", flush=True)
    return result


def _selection_context(protocol_path: Path, train_path: Path, key: str, replicate: dict, parts, code_identity: dict) -> dict:
    return {
        "protocol_sha256": _sha256_file(protocol_path), "train_sha256": _sha256_file(train_path),
        "dataset_key": key, "class_order_seed": replicate["class_order_seed"],
        "projection_seed": replicate["projection_seed"],
        "inner_fit_sha256": _sequence_sha256(parts[0]),
        "inner_validation_sha256": _sequence_sha256(parts[1]),
        "outer_fit_sha256": _sequence_sha256(parts[2]),
        "outer_validation_sha256": _sequence_sha256(parts[3]),
        "code_identity": code_identity,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
    }


def select_dataset(
    *, protocol_path: Path, dataset_key: str, feature_cache_dir: Path,
    code_cache_root: Path, output_root: Path, dataset_audit_path: Path | None,
    device_name: str,
) -> dict:
    protocol = _read_protocol(protocol_path); _verify_method_identity(protocol)
    dataset = protocol["datasets"][dataset_key]
    audit = _validate_dataset_audit(dataset_audit_path, dataset_key, dataset)
    train, _, metadata = _validate_train_cache(feature_cache_dir, protocol, dataset_key, require_test=False)
    selection, grid = protocol["selection"], list(map(float, protocol["selection"]["ridge_grid"]))
    output_dir = output_root / dataset_key; output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    replicate_data = []
    for replicate_index, replicate in enumerate(selection["development_replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(dataset["num_classes"]), dataset["num_classes"]
        )
        parts = _nested_parts(
            train["labels"], class_order, dataset["num_tasks"], selection["split_seed"],
            selection["outer_validation_fraction"], selection["inner_validation_fraction"],
        )
        cache = _prepare_code_cache(
            train=train, train_sha256=_sha256_file(feature_cache_dir / "train.pt"),
            cache_dir=code_cache_root / dataset_key / f"development_{replicate_index}",
            config=_cache_config(protocol, dataset, replicate["projection_seed"]), device=device_name,
        )
        context = _selection_context(
            protocol_path, feature_cache_dir / "train.pt", dataset_key, replicate, parts,
            {"identity_sha256": cache[2]["identity_sha256"], "projection_sha256": _tensor_content_sha256(cache[3])},
        )
        replicate_data.append({
            "replicate_index": replicate_index, "replicate": replicate,
            "class_order": class_order, "parts": parts, "cache": cache, "context": context,
        })

    fly_candidates, raw_candidates = [], []
    tolerance = selection["maximum_solver_relative_residual"]
    for candidate_index, ridge in enumerate(grid):
        fly_results, raw_results = [], []
        for item in replicate_data:
            base = {**item["context"], "candidate_index": candidate_index, "ridge_lambda": ridge}
            context_sha = _sha256_bytes(json.dumps(base, sort_keys=True).encode())
            config = _candidate_config(protocol, dataset, item["replicate"]["projection_seed"], ridge)
            inner_fit, inner_validation = item["parts"][0], item["parts"][1]
            fly = _run_unit(
                _unit_path(output_dir, f"inner_fly_r{item['replicate_index']}_c{candidate_index}"),
                context_sha, f"SELECT {dataset_key} fly-family rep={item['replicate_index']} lambda={ridge:g}",
                lambda config=config, item=item, inner_fit=inner_fit, inner_validation=inner_validation:
                    d1._evaluate_paired_exact_srq(
                        config=config, train=train, code_indices=item["cache"][0], code_values=item["cache"][1],
                        projection=item["cache"][3], training_parts=inner_fit,
                        validation_parts=inner_validation, device=device,
                    ),
            )
            raw_config = {**config, "statistics_dtype": protocol["representation"]["raw_statistics_dtype"]}
            raw = _run_unit(
                _unit_path(output_dir, f"inner_raw_r{item['replicate_index']}_c{candidate_index}"),
                context_sha, f"SELECT {dataset_key} raw rep={item['replicate_index']} lambda={ridge:g}",
                lambda raw_config=raw_config, inner_fit=inner_fit, inner_validation=inner_validation:
                    d0._evaluate_raw(
                        config=raw_config, train=train, training_parts=inner_fit,
                        validation_parts=inner_validation, device=device,
                    ),
            )
            fly_results.append(fly); raw_results.append(raw)
        valid_fly = all(
            result.get("status") == "complete"
            and max(result["exact"]["maximum_solver_relative_residual"], result["srq"]["maximum_solver_relative_residual"]) <= tolerance
            for result in fly_results
        )
        valid_raw = all(
            result.get("status") == "complete"
            and result.get("maximum_solver_relative_residual", float("inf")) <= tolerance
            for result in raw_results
        )
        fly_score = None if not valid_fly else statistics.fmean(
            (result["exact"]["validation_average_accuracy"] + result["srq"]["validation_average_accuracy"]) / 2
            for result in fly_results
        )
        raw_score = None if not valid_raw else statistics.fmean(
            result["validation_average_accuracy"] for result in raw_results
        )
        fly_candidates.append({
            "ridge_lambda": ridge, "valid": valid_fly, "mean_inner_aia": fly_score,
            "per_replicate": fly_results,
        })
        raw_candidates.append({
            "ridge_lambda": ridge, "valid": valid_raw, "mean_inner_aia": raw_score,
            "per_replicate": raw_results,
        })
    valid_fly = [item for item in fly_candidates if item["valid"]]
    valid_raw = [item for item in raw_candidates if item["valid"]]
    if not valid_fly or not valid_raw:
        raise RuntimeError("no numerically valid train-only hyperparameter candidate")
    selected_fly = max(valid_fly, key=lambda item: (item["mean_inner_aia"], item["ridge_lambda"]))
    selected_raw = max(valid_raw, key=lambda item: (item["mean_inner_aia"], item["ridge_lambda"]))
    boundary = selected_fly["ridge_lambda"] in {grid[0], grid[-1]} or selected_raw["ridge_lambda"] in {grid[0], grid[-1]}

    outer_results = []
    for item in replicate_data:
        outer_fit, outer_validation = item["parts"][2], item["parts"][3]
        base = {
            **item["context"], "selected_fly_lambda": selected_fly["ridge_lambda"],
            "selected_raw_lambda": selected_raw["ridge_lambda"], "phase": "outer_confirmation",
        }
        context_sha = _sha256_bytes(json.dumps(base, sort_keys=True).encode())
        fly_config = _candidate_config(
            protocol, dataset, item["replicate"]["projection_seed"], selected_fly["ridge_lambda"]
        )
        raw_config = {
            **_candidate_config(protocol, dataset, item["replicate"]["projection_seed"], selected_raw["ridge_lambda"]),
            "statistics_dtype": protocol["representation"]["raw_statistics_dtype"],
        }
        fly = _run_unit(
            _unit_path(output_dir, f"outer_fly_r{item['replicate_index']}"), context_sha,
            f"OUTER {dataset_key} fly-family rep={item['replicate_index']}",
            lambda fly_config=fly_config, item=item, outer_fit=outer_fit, outer_validation=outer_validation:
                d1._evaluate_paired_exact_srq(
                    config=fly_config, train=train, code_indices=item["cache"][0], code_values=item["cache"][1],
                    projection=item["cache"][3], training_parts=outer_fit,
                    validation_parts=outer_validation, device=device,
                ),
        )
        raw = _run_unit(
            _unit_path(output_dir, f"outer_raw_r{item['replicate_index']}"), context_sha,
            f"OUTER {dataset_key} raw rep={item['replicate_index']}",
            lambda raw_config=raw_config, outer_fit=outer_fit, outer_validation=outer_validation:
                d0._evaluate_raw(
                    config=raw_config, train=train, training_parts=outer_fit,
                    validation_parts=outer_validation, device=device,
                ),
        )
        outer_results.append({"replicate": item["replicate"], "fly_family": fly, "raw_ridge": raw})

    payload = {
        "schema_version": 1, "study_id": protocol["study_id"], "dataset_key": dataset_key,
        "status": "STOP_BOUNDARY_SELECTION" if boundary else "SELECTION_COMPLETE",
        "uses_test_set": False, "held_out_test_authorized": False,
        "protocol_sha256": _sha256_file(protocol_path), "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "selection_protocol": (
            "class-stratified 80/20 outer split, followed by an 80/20 inner split "
            "of the development partition, using official train only"
        ),
        "selection_metric": selection["fly_family_score"], "tie_break": selection["tie_break"],
        "grid": grid, "selected_fly_family_lambda": selected_fly["ridge_lambda"],
        "selected_raw_ridge_lambda": selected_raw["ridge_lambda"],
        "fly_candidates": fly_candidates, "raw_candidates": raw_candidates,
        "outer_confirmation": outer_results,
        "split_provenance": [
            {"replicate": item["replicate"], "class_order": item["class_order"], **{
                name: _sequence_sha256(parts) for name, parts in zip(
                    ("inner_fit_sha256", "inner_validation_sha256", "outer_fit_sha256", "outer_validation_sha256"),
                    item["parts"],
                )
            }} for item in replicate_data
        ],
        "source_feature_metadata": metadata, "dataset_audit": audit,
        "warning": "ImageNet-R is a disclosed legacy processed split with 19 cross-split duplicate hashes"
        if dataset_key == "imagenetr" else None,
    }
    _atomic_json(output_dir / "selection.json", payload)
    print(
        f"SELECTION COMPLETE dataset={dataset_key} status={payload['status']} "
        f"fly_lambda={selected_fly['ridge_lambda']:g} raw_lambda={selected_raw['ridge_lambda']:g}",
        flush=True,
    )
    return payload


def lock_selection(protocol_path: Path, selection_root: Path, output_root: Path, require_clean_git: bool) -> dict:
    protocol = _read_protocol(protocol_path); identities = _verify_method_identity(protocol)
    selections, selected = {}, {}
    for key in DATASET_KEYS:
        path = selection_root / key / "selection.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing train-only selection: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "SELECTION_COMPLETE"
            or payload.get("uses_test_set") is not False
            or payload.get("held_out_test_authorized") is not False
            or payload.get("protocol_sha256") != _sha256_file(protocol_path)
            or payload.get("runner_sha256") != _sha256_file(Path(__file__).resolve())
        ):
            raise ValueError(f"selection contract mismatch for {key}")
        grid = set(map(float, protocol["selection"]["ridge_grid"]))
        if (
            float(payload["selected_fly_family_lambda"]) not in grid
            or float(payload["selected_raw_ridge_lambda"]) not in grid
        ):
            raise ValueError(f"selected lambda is outside the locked grid for {key}")
        selections[key] = {"path": str(path), "sha256": _sha256_file(path)}
        selected[key] = {
            "fly_family_lambda": float(payload["selected_fly_family_lambda"]),
            "raw_ridge_lambda": float(payload["selected_raw_ridge_lambda"]),
        }
    git = _git_provenance()
    if require_clean_git and git.get("git_dirty") is not False:
        raise RuntimeError("final lock requires a clean Git worktree")
    record = {
        "schema_version": 1, "study_id": protocol["study_id"],
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _sha256_file(protocol_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "method_identity": identities, "selection_files": selections,
        "selected_hyperparameters": selected,
        "git_commit": git.get("git_commit"), "git_dirty": git.get("git_dirty"),
        "test_tuning_allowed": False, "accuracy_based_early_stop": False,
    }
    record["authorization_id"] = _sha256_bytes(json.dumps(record, sort_keys=True).encode())
    path = output_root / "authorization.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        immutable = ("protocol_sha256", "runner_sha256", "method_identity", "selection_files", "selected_hyperparameters", "git_commit")
        if any(previous.get(field) != record.get(field) for field in immutable):
            raise RuntimeError("existing authorization belongs to a different selection/code context")
        print(f"AUTHORIZATION RESTORED id={previous['authorization_id']}", flush=True)
        return previous
    _atomic_json(path, record)
    print(f"AUTHORIZATION LOCKED id={record['authorization_id']}", flush=True)
    return record


def _validate_authorization(path: Path, protocol_path: Path, selection_root: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError("authorization.json is missing")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("protocol_sha256") != _sha256_file(protocol_path)
        or record.get("runner_sha256") != _sha256_file(Path(__file__).resolve())
        or record.get("test_tuning_allowed") is not False
    ):
        raise ValueError("authorization source identity mismatch")
    for key in DATASET_KEYS:
        if record["selection_files"][key]["sha256"] != _sha256_file(selection_root / key / "selection.json"):
            raise ValueError(f"selection changed after authorization: {key}")
    return record


def extract_test(
    *, protocol_path: Path, dataset_key: str, authorization_path: Path,
    selection_root: Path, feature_cache_dir: Path, dataset_root: str,
    checkpoint_path: str, device_name: str, batch_size: int, num_workers: int,
) -> dict:
    protocol = _read_protocol(protocol_path)
    authorization = _validate_authorization(authorization_path, protocol_path, selection_root)
    dataset, backbone = protocol["datasets"][dataset_key], protocol["backbone"]
    test_path, metadata_path = feature_cache_dir / "test.pt", feature_cache_dir / "metadata.json"
    if test_path.exists():
        _, test, metadata = _validate_train_cache(feature_cache_dir, protocol, dataset_key, require_test=True)
        if metadata.get("authorization_id") != authorization["authorization_id"]:
            raise RuntimeError("existing test cache belongs to a different authorization")
        print(f"TEST CACHE RESTORED {dataset_key} shape={tuple(test['features'].shape)}", flush=True)
        return {"status": "restored", "test_sha256": _sha256_file(test_path)}
    _validate_train_cache(feature_cache_dir, protocol, dataset_key, require_test=False)
    random_initialization(protocol["selection"]["split_seed"])
    namespace = argparse.Namespace(
        dataset=dataset["dataset"], root=dataset_root, num_classes=dataset["num_classes"],
        num_tasks=dataset["num_tasks"], batch_size=batch_size,
        data_augmentation=backbone["preprocessing"], num_workers=num_workers,
    )
    _, test_loaders = load_dataset(namespace)
    device = torch.device(device_name)
    model = load_model(
        backbone["model_name"], checkpoint_path=checkpoint_path,
        expected_checkpoint_size=backbone["checkpoint_size"],
        expected_checkpoint_sha256=backbone["checkpoint_sha256"],
    ).eval().to(device)
    features, labels = [], []
    for task in range(dataset["num_tasks"]):
        values, targets = feature_extract(model, test_loaders[task], device)
        features.append(values.cpu()); labels.append(targets.cpu())
        print(f"TEST EXTRACT {dataset_key} task={task+1}/{dataset['num_tasks']} samples={len(targets)}", flush=True)
    packed = {"features": torch.cat(features), "labels": torch.cat(labels)}
    if (
        tuple(packed["features"].shape) != (dataset["test_samples"], backbone["feature_dim"])
        or tuple(packed["labels"].shape) != (dataset["test_samples"],)
        or not bool(torch.isfinite(packed["features"]).all())
    ):
        raise ValueError("held-out extraction tensor contract mismatch")
    _atomic_torch(test_path, packed)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "test_shape": list(packed["features"].shape), "test_labels_shape": list(packed["labels"].shape),
        "test_features_materialized": True, "authorization_id": authorization["authorization_id"],
        "test_sha256": _sha256_file(test_path),
    })
    _atomic_json(metadata_path, metadata)
    print(f"TEST CACHE COMPLETE {dataset_key} shape={tuple(packed['features'].shape)}", flush=True)
    return {"status": "complete", "test_sha256": metadata["test_sha256"]}


def _runtime_manifest(protocol: dict) -> dict:
    rep = protocol["representation"]
    return {
        "backbone": protocol["backbone"],
        "representation": {
            "large_expand_dim": rep["expand_dim"], "synaptic_degree": rep["synaptic_degree"],
            "coding_level": rep["coding_level"], "encode_batch_size": rep["encode_batch_size"],
            "evaluation_batch_size": rep["evaluation_batch_size"], "block_size": rep["block_size"],
            "group_size": rep["group_size"], "statistics_dtype": rep["statistics_dtype"],
            "solver_dtype": rep["solver_dtype"], "raw_statistics_dtype": rep["raw_statistics_dtype"],
        },
        "reporting": {"maximum_solver_relative_residual": protocol["selection"]["maximum_solver_relative_residual"]},
    }


def _heldout_unit(path: Path, context_sha: str, label: str, evaluator) -> dict:
    result = _load_unit(path, context_sha)
    if result is not None:
        print(f"RESTORED {label}", flush=True); return result
    print(f"START {label}", flush=True)
    try:
        result = evaluator()
    except (RuntimeError, torch.linalg.LinAlgError) as error:
        result = {"status": "numerical_failure", "uses_test_set": True, "failure": f"{type(error).__name__}: {error}"}
    result = _save_unit(path, context_sha, result)
    print(f"DONE {label} status={result['status']}", flush=True)
    return result


def evaluate_dataset(
    *, protocol_path: Path, dataset_key: str, selection_root: Path,
    authorization_path: Path, feature_cache_dir: Path, code_cache_root: Path,
    output_root: Path, dataset_audit_path: Path | None, device_name: str,
) -> dict:
    protocol = _read_protocol(protocol_path); _verify_method_identity(protocol)
    authorization = _validate_authorization(authorization_path, protocol_path, selection_root)
    dataset_static = protocol["datasets"][dataset_key]
    audit = _validate_dataset_audit(dataset_audit_path, dataset_key, dataset_static)
    train, test, metadata = _validate_train_cache(feature_cache_dir, protocol, dataset_key, require_test=True)
    selection = json.loads((selection_root / dataset_key / "selection.json").read_text(encoding="utf-8"))
    dataset = {
        **dataset_static,
        "fly_ridge_lambda": float(selection["selected_fly_family_lambda"]),
        "raw_ridge_lambda": float(selection["selected_raw_ridge_lambda"]),
    }
    manifest = _runtime_manifest(protocol)
    stream = {"features": torch.cat((train["features"], test["features"])), "labels": torch.cat((train["labels"], test["labels"]))}
    offset, device = len(train["labels"]), torch.device(device_name)
    source_sha = _sha256_bytes((_sha256_file(feature_cache_dir / "train.pt") + _sha256_file(feature_cache_dir / "test.pt")).encode())
    dataset_output = output_root / dataset_key; dataset_output.mkdir(parents=True, exist_ok=True)
    seed_results = []
    for replicate_index, replicate in enumerate(protocol["final_evaluation"]["replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(range(dataset["num_classes"]), dataset["num_classes"])
        classes_per_task = dataset["num_classes"] // dataset["num_tasks"]
        train_parts, test_parts = [], []
        for task in range(dataset["num_tasks"]):
            ids = torch.tensor(class_order[task * classes_per_task:(task + 1) * classes_per_task])
            train_parts.append(torch.nonzero(torch.isin(train["labels"], ids)).flatten())
            test_parts.append(torch.nonzero(torch.isin(test["labels"], ids)).flatten() + offset)
        cache = _prepare_code_cache(
            train=stream, train_sha256=source_sha,
            cache_dir=code_cache_root / dataset_key / f"replicate_{replicate_index}",
            config=_cache_config(protocol, dataset, replicate["projection_seed"]), device=device_name,
        )
        context = {
            "protocol_sha256": _sha256_file(protocol_path), "authorization_id": authorization["authorization_id"],
            "selection_sha256": _sha256_file(selection_root / dataset_key / "selection.json"),
            "dataset_key": dataset_key, "replicate": replicate, "class_order": class_order,
            "train_indices_sha256": _sequence_sha256(train_parts), "test_indices_sha256": _sequence_sha256(test_parts),
            "code_identity": cache[2]["identity_sha256"], "projection_sha256": _tensor_content_sha256(cache[3]),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
        }
        context_sha = _sha256_bytes(json.dumps(context, sort_keys=True).encode())
        paired = _heldout_unit(
            _unit_path(dataset_output, f"replicate_{replicate_index}_paired"), context_sha,
            f"HELDOUT {dataset_key} replicate={replicate_index} paired",
            lambda cache=cache, train_parts=train_parts, test_parts=test_parts, replicate=replicate:
                _evaluate_paired(
                    manifest=manifest, dataset=dataset, seed=replicate["projection_seed"], stream=stream,
                    code_indices=cache[0], code_values=cache[1], projection=cache[3],
                    training_parts=train_parts, test_parts=test_parts, device=device,
                ),
        )
        raw = _heldout_unit(
            _unit_path(dataset_output, f"replicate_{replicate_index}_raw"), context_sha,
            f"HELDOUT {dataset_key} replicate={replicate_index} raw",
            lambda train_parts=train_parts, test_parts=test_parts, replicate=replicate:
                _evaluate_raw(
                    manifest=manifest, dataset=dataset, seed=replicate["class_order_seed"], stream=stream,
                    training_parts=train_parts, test_parts=test_parts, device=device,
                ),
        )
        methods = {
            "exact_fly_10000": paired.get("exact", paired),
            "srq_fly_10000": paired.get("srq", paired),
            "raw_ridge": raw,
        }
        seed_results.append({
            "replicate_index": replicate_index, "class_order_seed": replicate["class_order_seed"],
            "projection_seed": replicate["projection_seed"], "class_order": class_order,
            "methods": methods,
            "paired_diagnostics": {
                "prediction_agreement_by_task": paired.get("prediction_agreement_by_task"),
                "relative_logit_frobenius_error_by_task": paired.get("relative_logit_frobenius_error_by_task"),
            },
        })
        print(f"REPLICATE COMPLETE dataset={dataset_key} index={replicate_index}", flush=True)
    failures = [
        {"replicate_index": item["replicate_index"], "method": method, "status": result.get("status")}
        for item in seed_results for method, result in item["methods"].items()
        if result.get("status") != "complete"
    ]
    payload = {
        "schema_version": 1, "study_id": protocol["study_id"], "dataset_key": dataset_key,
        "status": "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
        "uses_test_set": True, "test_tuning_allowed": False, "accuracy_based_early_stop": False,
        "authorization_id": authorization["authorization_id"], "selected_hyperparameters": {
            "fly_family_lambda": dataset["fly_ridge_lambda"], "raw_ridge_lambda": dataset["raw_ridge_lambda"]
        },
        "source_feature_metadata": metadata, "dataset_audit": audit,
        "exact_command": [sys.executable, *sys.argv], "environment": _environment(device_name),
        "feature_cache_disk_bytes": sum(path.stat().st_size for path in feature_cache_dir.iterdir() if path.is_file()),
        "wta_cache_disk_bytes": sum(path.stat().st_size for path in (code_cache_root / dataset_key).rglob("*") if path.is_file()),
        "legacy_processed_split_disclosure": "19 cross-split duplicate hashes, including 18 conflicting-label hashes; not content-disjoint"
        if dataset_key == "imagenetr" else None,
        "seed_results": seed_results, "failures": failures,
    }
    _atomic_json(dataset_output / "heldout_results.json", payload)
    print(f"DATASET COMPLETE {dataset_key} status={payload['status']}", flush=True)
    return payload


def summarize(protocol_path: Path, output_root: Path) -> dict:
    protocol = _read_protocol(protocol_path)
    rows, curves, paired = [], [], {}
    summaries = {}
    for key in DATASET_KEYS:
        path = output_root / key / "heldout_results.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing held-out result: {key}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("uses_test_set") is not True or payload.get("test_tuning_allowed") is not False:
            raise ValueError(f"invalid held-out result contract: {key}")
        summaries[key] = {}
        for method in METHODS:
            results = [item["methods"][method] for item in payload["seed_results"]]
            if any(result.get("status") != "complete" for result in results):
                summaries[key][method] = {"status": "incomplete"}
                rows.append({"dataset": key, "method": method, "status": "incomplete"})
                continue
            metrics = {}
            for metric in (
                "final_accuracy", "average_incremental_accuracy", "forgetting",
                "persistent_state_bytes", "total_update_seconds", "total_inference_seconds",
            ):
                metrics[metric] = _mean_std_ci([float(result[metric]) for result in results])
            summaries[key][method] = {"status": "complete", **metrics}
            row = {"dataset": key, "method": method, "status": "complete"}
            for metric, values in metrics.items():
                for field in ("mean", "sample_std", "ci95_low", "ci95_high"):
                    row[f"{metric}_{field}"] = values[field]
            rows.append(row)
            for replicate_index, result in enumerate(results):
                for task, value in enumerate(result["stage_accuracy"], 1):
                    curves.append({
                        "dataset": key, "method": method, "replicate_index": replicate_index,
                        "task": task, "task_fraction": task / len(result["stage_accuracy"]),
                        "average_seen_accuracy": value,
                    })
        differences = []
        for item in payload["seed_results"]:
            exact, srq = item["methods"]["exact_fly_10000"], item["methods"]["srq_fly_10000"]
            if exact.get("status") == srq.get("status") == "complete":
                differences.append(srq["average_incremental_accuracy"] - exact["average_incremental_accuracy"])
        paired[key] = _mean_std_ci(differences) if differences else {"n": 0}
    summary = {
        "schema_version": 1, "study_id": protocol["study_id"],
        "status": "REPORTED_WITHOUT_ACCURACY_GATE", "uses_test_set": True,
        "protocol_sha256": _sha256_file(protocol_path), "dataset_method_summaries": summaries,
        "paired_srq_minus_exact_fly_aia": paired,
        "imagenetr_disclosure": "legacy processed split with 19 cross-split duplicate hashes; not content-disjoint",
    }
    _atomic_json(output_root / "final_summary.json", summary)
    for path, data in ((output_root / "metrics_summary.csv", rows), (output_root / "task_curves.csv", curves)):
        fields = sorted({field for row in data for field in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(data)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--protocol", required=True); select.add_argument("--dataset-key", choices=DATASET_KEYS, required=True)
    select.add_argument("--feature-cache-dir", required=True); select.add_argument("--code-cache-root", required=True)
    select.add_argument("--output-root", required=True); select.add_argument("--dataset-audit"); select.add_argument("--device", default="cpu")
    lock = sub.add_parser("lock")
    lock.add_argument("--protocol", required=True); lock.add_argument("--selection-root", required=True)
    lock.add_argument("--output-root", required=True); lock.add_argument("--require-clean-git", action="store_true")
    extract = sub.add_parser("extract-test")
    extract.add_argument("--protocol", required=True); extract.add_argument("--dataset-key", choices=DATASET_KEYS, required=True)
    extract.add_argument("--selection-root", required=True); extract.add_argument("--authorization", required=True)
    extract.add_argument("--feature-cache-dir", required=True); extract.add_argument("--root", required=True)
    extract.add_argument("--backbone-checkpoint", required=True); extract.add_argument("--device", default="cpu")
    extract.add_argument("--batch-size", type=int, default=128); extract.add_argument("--num-workers", type=int, default=2)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--protocol", required=True); evaluate.add_argument("--dataset-key", choices=DATASET_KEYS, required=True)
    evaluate.add_argument("--selection-root", required=True); evaluate.add_argument("--authorization", required=True)
    evaluate.add_argument("--feature-cache-dir", required=True); evaluate.add_argument("--code-cache-root", required=True)
    evaluate.add_argument("--output-root", required=True); evaluate.add_argument("--dataset-audit"); evaluate.add_argument("--device", default="cpu")
    report = sub.add_parser("summarize")
    report.add_argument("--protocol", required=True); report.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv); protocol_path = Path(args.protocol).resolve()
    if args.command == "select":
        select_dataset(
            protocol_path=protocol_path, dataset_key=args.dataset_key,
            feature_cache_dir=Path(args.feature_cache_dir).resolve(), code_cache_root=Path(args.code_cache_root).resolve(),
            output_root=Path(args.output_root).resolve(),
            dataset_audit_path=None if args.dataset_audit is None else Path(args.dataset_audit).resolve(),
            device_name=args.device,
        )
    elif args.command == "lock":
        lock_selection(
            protocol_path, Path(args.selection_root).resolve(), Path(args.output_root).resolve(), args.require_clean_git
        )
    elif args.command == "extract-test":
        extract_test(
            protocol_path=protocol_path, dataset_key=args.dataset_key,
            authorization_path=Path(args.authorization).resolve(), selection_root=Path(args.selection_root).resolve(),
            feature_cache_dir=Path(args.feature_cache_dir).resolve(), dataset_root=args.root,
            checkpoint_path=args.backbone_checkpoint, device_name=args.device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
    elif args.command == "evaluate":
        evaluate_dataset(
            protocol_path=protocol_path, dataset_key=args.dataset_key,
            selection_root=Path(args.selection_root).resolve(), authorization_path=Path(args.authorization).resolve(),
            feature_cache_dir=Path(args.feature_cache_dir).resolve(), code_cache_root=Path(args.code_cache_root).resolve(),
            output_root=Path(args.output_root).resolve(),
            dataset_audit_path=None if args.dataset_audit is None else Path(args.dataset_audit).resolve(),
            device_name=args.device,
        )
    else:
        summarize(protocol_path, Path(args.output_root).resolve())


if __name__ == "__main__":
    main()
