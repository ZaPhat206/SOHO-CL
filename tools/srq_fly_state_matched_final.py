"""State-matched Exact-FLY control for the locked SRQ-FLY P2B result.

Width is selected solely from deployed persistent tensor bytes.  Ridge is then
selected on nested partitions of the official training split.  Test evaluation
is unavailable until all three selections, the immutable P2B result artifact,
and the current clean Git commit have been bound into an authorization record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import srq_fly_d0 as d0
from tools import srq_fly_selfcontained as base
from tools.srq_fly_d2_state_match import exact_fly_state_bytes
from tools.srq_fly_heldout import (
    _evaluate_exact_matched,
    _mean_std_ci,
)
from tools.tail_fly_phasea import _load_unit, _save_unit, _unit_path
from tools.twa_fly_pilot import (
    _prepare_code_cache,
    _sequence_sha256,
    _sha256_bytes,
    _sha256_file,
    _tensor_content_sha256,
)


DATASET_KEYS = ("cifar100", "cub200", "imagenetr")
REFERENCE_METHODS = (
    "exact_fly_10000", "srq_fly_p2b_10000", "raw_ridge",
)


def _source_identity() -> dict[str, str]:
    return {
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "optimized_learner_sha256": _sha256_file(
            ROOT / "methods/srq_fly_optimized/learner.py"
        ),
        "optimized_storage_sha256": _sha256_file(
            ROOT / "methods/srq_fly_optimized/storage.py"
        ),
        "flyhash_sha256": _sha256_file(ROOT / "models/flyhash.py"),
        "heldout_helper_sha256": _sha256_file(ROOT / "tools/srq_fly_heldout.py"),
        "code_cache_helper_sha256": _sha256_file(ROOT / "tools/twa_fly_pilot.py"),
    }


def closest_non_exceeding_width(
    *, target_bytes: int, feature_dim: int, synaptic_degree: int,
    num_classes: int, maximum_width: int,
) -> dict[str, int | float]:
    """Choose width by bytes only; accuracy is intentionally not an input."""
    if min(target_bytes, feature_dim, synaptic_degree, num_classes, maximum_width) <= 0:
        raise ValueError("state-match inputs must be positive")
    lo, hi, answer = 1, maximum_width, None
    while lo <= hi:
        middle = (lo + hi) // 2
        state = exact_fly_state_bytes(
            feature_dim=feature_dim, expand_dim=middle,
            synaptic_degree=synaptic_degree, num_classes=num_classes,
        )
        if state <= target_bytes:
            answer = (middle, state)
            lo = middle + 1
        else:
            hi = middle - 1
    if answer is None:
        raise ValueError("target is smaller than width-one Exact-FLY state")
    width, state = answer
    return {
        "width": width,
        "exact_fly_state_bytes": state,
        "target_p2b_state_bytes": target_bytes,
        "relative_byte_gap": (target_bytes - state) / target_bytes,
    }


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "study_id", "base_protocol", "p2b_reference",
        "method_identity", "state_matching", "selection", "final_evaluation",
        "datasets",
    }
    if set(config) != required or config.get("schema_version") != 1:
        raise ValueError("state-matched final config schema mismatch")
    if set(config["datasets"]) != set(DATASET_KEYS):
        raise ValueError("state-matched protocol must cover three datasets")
    source = _source_identity().copy()
    source.pop("runner_sha256")
    if source != config["method_identity"]:
        raise ValueError("state-matched source identity mismatch")
    base_path = ROOT / config["base_protocol"]["path"]
    if _sha256_file(base_path) != config["base_protocol"]["sha256"]:
        raise ValueError("base protocol identity mismatch")
    if _sha256_file(ROOT / config["base_protocol"]["runner_path"]) != config["base_protocol"]["runner_sha256"]:
        raise ValueError("base runner identity mismatch")
    base_protocol = base._read_protocol(base_path)
    state = config["state_matching"]
    if (
        state["large_width"] != 10000
        or state["feature_dim"] != base_protocol["backbone"]["feature_dim"]
        or state["synaptic_degree"] != base_protocol["representation"]["synaptic_degree"]
        or state["coding_level"] != base_protocol["representation"]["coding_level"]
        or state["statistics_dtype"] != "float32"
        or not 0 < state["maximum_relative_byte_gap"] <= 0.001
    ):
        raise ValueError("state-match representation contract mismatch")
    selection = config["selection"]
    base_selection = base_protocol["selection"]
    for key in (
        "split_seed", "outer_validation_fraction", "inner_validation_fraction",
        "ridge_grid", "development_replicates",
        "maximum_solver_relative_residual",
    ):
        if selection[key] != base_selection[key]:
            raise ValueError(f"selection protocol diverged from base: {key}")
    if selection["split_seed"] != 2025 or len(selection["development_replicates"]) != 3:
        raise ValueError("invalid development protocol")
    final = config["final_evaluation"]
    if (
        final["replicates"] != base_protocol["final_evaluation"]["replicates"]
        or len(final["replicates"]) != 6
        or final["test_tuning_allowed"] is not False
        or final["accuracy_based_early_stop"] is not False
    ):
        raise ValueError("final replicate contract mismatch")
    for key, dataset in config["datasets"].items():
        if dataset != base_protocol["datasets"][key]:
            raise ValueError(f"dataset protocol diverged from base: {key}")
        match = closest_non_exceeding_width(
            target_bytes=int(state["p2b_target_bytes"][key]),
            feature_dim=int(state["feature_dim"]),
            synaptic_degree=int(state["synaptic_degree"]),
            num_classes=int(dataset["num_classes"]),
            maximum_width=int(state["large_width"] - 1),
        )
        if match["width"] != state["selected_widths"][key]:
            raise ValueError(f"selected width is not byte-derived for {key}")
        if match["relative_byte_gap"] > state["maximum_relative_byte_gap"]:
            raise ValueError(f"state match exceeds byte-gap tolerance for {key}")
    return config


def _base_protocol(config: dict) -> dict:
    return base._read_protocol(ROOT / config["base_protocol"]["path"])


def _representation(config: dict, dataset_key: str) -> dict:
    protocol = _base_protocol(config)
    rep = protocol["representation"]
    return {
        "expand_dim": int(config["state_matching"]["selected_widths"][dataset_key]),
        "synaptic_degree": rep["synaptic_degree"],
        "coding_level": rep["coding_level"],
        "encode_batch_size": rep["encode_batch_size"],
        "evaluation_batch_size": rep["evaluation_batch_size"],
    }


def _cache_config(config: dict, dataset_key: str, projection_seed: int) -> dict:
    protocol = _base_protocol(config)
    return {
        "seed": projection_seed,
        "num_classes": config["datasets"][dataset_key]["num_classes"],
        "representation": _representation(config, dataset_key),
        "statistics_dtype": "float32",
        "raw_ridge_lambda": 1.0,
        "solver_tolerance": config["selection"]["maximum_solver_relative_residual"],
        "solver_max_iterations": 100,
    }


def _candidate_config(config: dict, dataset_key: str, seed: int, ridge: float) -> dict:
    protocol = _base_protocol(config)
    representation = _representation(config, dataset_key)
    return {
        "seed": seed,
        "ridge_lambda": ridge,
        "raw_ridge_lambda": ridge,
        "statistics_dtype": "float32",
        "solver_dtype": "float32",
        "large_representation": {
            "expand_dim": representation["expand_dim"],
            "synaptic_degree": representation["synaptic_degree"],
            "coding_level": representation["coding_level"],
            "evaluation_batch_size": representation["evaluation_batch_size"],
        },
        "storage": {
            "block_size": protocol["representation"]["block_size"],
            "group_size": protocol["representation"]["group_size"],
        },
    }


def _run_train_unit(path: Path, context_sha: str, label: str, evaluator) -> dict:
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
            "status": "numerical_failure", "uses_test_set": False,
            "failure": f"{type(error).__name__}: {error}",
        }
    result["unit_seconds"] = time.perf_counter() - started
    result = _save_unit(path, context_sha, result)
    print(f"DONE {label} status={result['status']}", flush=True)
    return result


def _evaluate_train_exact(
    *, config: dict, dataset_key: str, ridge: float, seed: int, train: dict,
    cache: tuple, fit_parts: list[torch.Tensor], validation_parts: list[torch.Tensor],
    device: torch.device,
) -> dict:
    representation = _representation(config, dataset_key)
    return d0._evaluate_exact(
        name="exact_fly_state_matched",
        config=_candidate_config(config, dataset_key, seed, ridge),
        representation=representation,
        train=train, code_indices=cache[0], code_values=cache[1],
        projection=cache[3], training_parts=fit_parts,
        validation_parts=validation_parts, device=device,
    )


def select_dataset(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    protocol = _base_protocol(config)
    key = args.dataset_key
    dataset = config["datasets"][key]
    audit = base._validate_dataset_audit(
        None if args.dataset_audit is None else Path(args.dataset_audit).resolve(),
        key, dataset,
    )
    feature_cache = Path(args.feature_cache_dir).resolve()
    train, _, metadata = base._validate_train_cache(
        feature_cache, protocol, key, require_test=False
    )
    train_sha = _sha256_file(feature_cache / "train.pt")
    output = Path(args.output_root).resolve() / key
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    selection = config["selection"]
    replicate_data = []
    for index, replicate in enumerate(selection["development_replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(dataset["num_classes"]), dataset["num_classes"]
        )
        parts = base._nested_parts(
            train["labels"], class_order, dataset["num_tasks"],
            selection["split_seed"], selection["outer_validation_fraction"],
            selection["inner_validation_fraction"],
        )
        cache = _prepare_code_cache(
            train=train, train_sha256=train_sha,
            cache_dir=Path(args.code_cache_root).resolve() / key / f"development_{index}",
            config=_cache_config(config, key, replicate["projection_seed"]),
            device=args.device,
        )
        context = {
            "config_sha256": _sha256_file(config_path), "dataset_key": key,
            "train_sha256": train_sha, "replicate": replicate,
            "class_order": class_order,
            "inner_fit_sha256": _sequence_sha256(parts[0]),
            "inner_validation_sha256": _sequence_sha256(parts[1]),
            "outer_fit_sha256": _sequence_sha256(parts[2]),
            "outer_validation_sha256": _sequence_sha256(parts[3]),
            "code_identity": cache[2]["identity_sha256"],
            "projection_sha256": _tensor_content_sha256(cache[3]),
            "runner_sha256": _source_identity()["runner_sha256"],
        }
        replicate_data.append((index, replicate, class_order, parts, cache, context))

    candidates = []
    tolerance = float(selection["maximum_solver_relative_residual"])
    for candidate_index, ridge in enumerate(map(float, selection["ridge_grid"])):
        results = []
        for index, replicate, _, parts, cache, context in replicate_data:
            unit_context = {**context, "candidate_index": candidate_index, "ridge_lambda": ridge}
            context_sha = _sha256_bytes(json.dumps(unit_context, sort_keys=True).encode())
            result = _run_train_unit(
                _unit_path(output, f"inner_matched_r{index}_c{candidate_index}"),
                context_sha,
                f"SELECT {key} matched rep={index} lambda={ridge:g}",
                lambda ridge=ridge, replicate=replicate, parts=parts, cache=cache:
                    _evaluate_train_exact(
                        config=config, dataset_key=key, ridge=ridge,
                        seed=replicate["projection_seed"], train=train, cache=cache,
                        fit_parts=parts[0], validation_parts=parts[1], device=device,
                    ),
            )
            results.append(result)
        valid = all(
            item.get("status") == "complete"
            and item.get("maximum_solver_relative_residual", float("inf")) <= tolerance
            for item in results
        )
        candidates.append({
            "ridge_lambda": ridge, "valid": valid,
            "mean_inner_aia": statistics.fmean(
                item["validation_average_accuracy"] for item in results
            ) if valid else None,
            "per_replicate": results,
        })
    valid = [item for item in candidates if item["valid"]]
    if not valid:
        raise RuntimeError("no numerically valid state-matched Ridge candidate")
    selected = max(valid, key=lambda item: (item["mean_inner_aia"], item["ridge_lambda"]))
    grid = list(map(float, selection["ridge_grid"]))
    boundary = selected["ridge_lambda"] in {grid[0], grid[-1]}
    outer = []
    for index, replicate, _, parts, cache, context in replicate_data:
        unit_context = {
            **context, "phase": "outer_confirmation",
            "selected_ridge_lambda": selected["ridge_lambda"],
        }
        result = _run_train_unit(
            _unit_path(output, f"outer_matched_r{index}"),
            _sha256_bytes(json.dumps(unit_context, sort_keys=True).encode()),
            f"OUTER {key} matched rep={index}",
            lambda replicate=replicate, parts=parts, cache=cache:
                _evaluate_train_exact(
                    config=config, dataset_key=key,
                    ridge=selected["ridge_lambda"],
                    seed=replicate["projection_seed"], train=train, cache=cache,
                    fit_parts=parts[2], validation_parts=parts[3], device=device,
                ),
        )
        outer.append({"replicate": replicate, "result": result})
    outer_valid = all(
        item["result"].get("status") == "complete"
        and item["result"].get(
            "maximum_solver_relative_residual", float("inf")
        ) <= tolerance
        for item in outer
    )
    match = closest_non_exceeding_width(
        target_bytes=int(config["state_matching"]["p2b_target_bytes"][key]),
        feature_dim=config["state_matching"]["feature_dim"],
        synaptic_degree=config["state_matching"]["synaptic_degree"],
        num_classes=dataset["num_classes"],
        maximum_width=config["state_matching"]["large_width"] - 1,
    )
    status = (
        "STOP_BOUNDARY_SELECTION" if boundary
        else "STOP_OUTER_NUMERICAL" if not outer_valid
        else "SELECTION_COMPLETE"
    )
    payload = {
        "schema_version": 1, "study_id": config["study_id"],
        "dataset_key": key,
        "status": status,
        "uses_test_set": False, "held_out_test_authorized": False,
        "config_sha256": _sha256_file(config_path),
        "runner_sha256": _source_identity()["runner_sha256"],
        "state_match": match,
        "selection_protocol": (
            "class-stratified nested 80/20 outer and 80/20 inner split of "
            "official train only; width fixed by bytes before accuracy"
        ),
        "grid": grid, "score": selection["score"],
        "tie_break": selection["tie_break"],
        "selected_ridge_lambda": selected["ridge_lambda"],
        "candidates": candidates, "outer_confirmation": outer,
        "split_provenance": [
            {
                "replicate": replicate, "class_order": class_order,
                "inner_fit_sha256": _sequence_sha256(parts[0]),
                "inner_validation_sha256": _sequence_sha256(parts[1]),
                "outer_fit_sha256": _sequence_sha256(parts[2]),
                "outer_validation_sha256": _sequence_sha256(parts[3]),
            }
            for _, replicate, class_order, parts, _, _ in replicate_data
        ],
        "source_feature_metadata": metadata, "dataset_audit": audit,
        "warning": (
            "ImageNet-R is the disclosed legacy processed split with 19 "
            "cross-split duplicate hashes" if key == "imagenetr" else None
        ),
    }
    base._atomic_json(output / "selection.json", payload)
    print(
        f"SELECTION COMPLETE dataset={key} status={payload['status']} "
        f"width={match['width']} lambda={selected['ridge_lambda']:g} "
        f"state_gap={100*match['relative_byte_gap']:.5f}%",
        flush=True,
    )
    return payload


def _reference_members(config: dict) -> dict[str, str]:
    reference = config["p2b_reference"]
    return {
        key: reference["results_member_template"].format(dataset_key=key)
        for key in DATASET_KEYS
    }


def read_reference_artifact(config: dict, artifact_path: Path) -> dict:
    reference = config["p2b_reference"]
    if not artifact_path.is_file():
        raise FileNotFoundError(f"missing immutable P2B artifact: {artifact_path}")
    if _sha256_file(artifact_path) != reference["artifact_sha256"]:
        raise ValueError("P2B reference artifact SHA-256 mismatch")
    with zipfile.ZipFile(artifact_path) as archive:
        summary_bytes = archive.read(reference["summary_member"])
        if hashlib.sha256(summary_bytes).hexdigest() != reference["summary_sha256"]:
            raise ValueError("P2B reference summary SHA-256 mismatch")
        summary = json.loads(summary_bytes)
        results = {
            key: json.loads(archive.read(member))
            for key, member in _reference_members(config).items()
        }
    if (
        summary.get("study_id") != reference["study_id"]
        or summary.get("status") != "CONFIRMATION_REPORTED_WITHOUT_ACCURACY_GATE"
        or summary.get("uses_test_set") is not True
        or summary.get("test_tuning_allowed") is not False
    ):
        raise ValueError("P2B reference summary contract mismatch")
    expected_replicates = config["final_evaluation"]["replicates"]
    for key, payload in results.items():
        if (
            payload.get("study_id") != reference["study_id"]
            or payload.get("dataset_key") != key
            or payload.get("status") != "CONFIRMATION_COMPLETE"
            or payload.get("uses_test_set") is not True
            or payload.get("test_tuning_allowed") is not False
            or len(payload.get("seed_results", [])) != 6
        ):
            raise ValueError(f"P2B reference result contract mismatch: {key}")
        observed_replicates = [
            {
                "class_order_seed": row["class_order_seed"],
                "projection_seed": row["projection_seed"],
            }
            for row in payload["seed_results"]
        ]
        if observed_replicates != expected_replicates:
            raise ValueError(f"P2B reference replicate mismatch: {key}")
        for row in payload["seed_results"]:
            if set(row.get("methods", {})) != set(REFERENCE_METHODS):
                raise ValueError(f"P2B reference method mismatch: {key}")
    return {"summary": summary, "results": results}


def _validate_selections(config_path: Path, config: dict, selection_root: Path) -> dict:
    selections = {}
    for key in DATASET_KEYS:
        path = selection_root / key / "selection.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing train-only selection: {key}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_match = closest_non_exceeding_width(
            target_bytes=config["state_matching"]["p2b_target_bytes"][key],
            feature_dim=config["state_matching"]["feature_dim"],
            synaptic_degree=config["state_matching"]["synaptic_degree"],
            num_classes=config["datasets"][key]["num_classes"],
            maximum_width=config["state_matching"]["large_width"] - 1,
        )
        if (
            payload.get("status") != "SELECTION_COMPLETE"
            or payload.get("uses_test_set") is not False
            or payload.get("held_out_test_authorized") is not False
            or payload.get("config_sha256") != _sha256_file(config_path)
            or payload.get("runner_sha256") != _source_identity()["runner_sha256"]
            or payload.get("state_match") != expected_match
            or payload.get("grid") != list(map(float, config["selection"]["ridge_grid"]))
            or float(payload.get("selected_ridge_lambda", 0)) <= 0
        ):
            raise ValueError(f"invalid state-matched selection: {key}")
        selections[key] = {
            "path": str(path), "sha256": _sha256_file(path),
            "width": expected_match["width"],
            "ridge_lambda": float(payload["selected_ridge_lambda"]),
            "state_match": expected_match,
        }
    return selections


def lock_confirmation(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    selection_root = Path(args.selection_root).resolve()
    selections = _validate_selections(config_path, config, selection_root)
    artifact = Path(args.reference_artifact).resolve()
    read_reference_artifact(config, artifact)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if args.require_clean_git and dirty:
        raise RuntimeError(f"state-matched lock requires a clean Git worktree:\n{dirty}")
    record = {
        "schema_version": 1, "study_id": config["study_id"],
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": _sha256_file(config_path),
        "source_identity": _source_identity(),
        "reference_artifact_sha256": config["p2b_reference"]["artifact_sha256"],
        "selection_sha256": {key: value["sha256"] for key, value in selections.items()},
        "selected_hyperparameters": {
            key: {"width": value["width"], "ridge_lambda": value["ridge_lambda"]}
            for key, value in selections.items()
        },
        "state_match": {key: value["state_match"] for key, value in selections.items()},
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(dirty), "test_tuning_allowed": False,
        "accuracy_based_early_stop": False,
        "prior_test_use_disclosure": config["p2b_reference"]["prior_test_use_disclosure"],
    }
    record["authorization_id"] = _sha256_bytes(
        json.dumps(record, sort_keys=True).encode()
    )
    output = Path(args.output_root).resolve() / "state_matched_authorization.json"
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        immutable = (
            "config_sha256", "source_identity", "reference_artifact_sha256",
            "selection_sha256", "selected_hyperparameters", "state_match",
            "git_commit", "git_dirty", "test_tuning_allowed",
            "accuracy_based_early_stop",
        )
        if any(previous.get(key) != record.get(key) for key in immutable):
            raise RuntimeError("existing state-matched authorization context mismatch")
        print(f"AUTHORIZATION RESTORED id={previous['authorization_id']}", flush=True)
        return previous
    base._atomic_json(output, record)
    print(f"AUTHORIZATION LOCKED id={record['authorization_id']}", flush=True)
    return record


def _validate_authorization(
    *, path: Path, config_path: Path, config: dict, selection_root: Path,
    reference_artifact: Path,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError("state-matched authorization is missing")
    record = json.loads(path.read_text(encoding="utf-8"))
    claimed = record.get("authorization_id")
    identity = dict(record)
    identity.pop("authorization_id", None)
    selections = _validate_selections(config_path, config, selection_root)
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if (
        record.get("config_sha256") != _sha256_file(config_path)
        or record.get("source_identity") != _source_identity()
        or record.get("reference_artifact_sha256") != _sha256_file(reference_artifact)
        or record.get("selection_sha256")
        != {key: value["sha256"] for key, value in selections.items()}
        or record.get("git_commit") != current_commit
        or record.get("git_dirty") is not False
        or bool(dirty)
        or claimed != _sha256_bytes(json.dumps(identity, sort_keys=True).encode())
        or record.get("test_tuning_allowed") is not False
        or record.get("accuracy_based_early_stop") is not False
    ):
        raise ValueError("state-matched authorization mismatch")
    read_reference_artifact(config, reference_artifact)
    return record


def _ordered_task_loaders(loaders, expected_tasks: int):
    """Validate and order the dictionary returned by ``load_dataset``."""
    if not isinstance(loaders, dict):
        raise TypeError("test loaders must be a task-indexed dictionary")
    expected_keys = set(range(expected_tasks))
    if set(loaders) != expected_keys:
        raise ValueError(
            "test loader task IDs mismatch: "
            f"expected {sorted(expected_keys)}, got {sorted(loaders)}"
        )
    return [loaders[index] for index in range(expected_tasks)]


def extract_test(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    protocol = _base_protocol(config)
    key = args.dataset_key
    feature_cache = Path(args.feature_cache_dir).resolve()
    authorization = _validate_authorization(
        path=Path(args.authorization).resolve(), config_path=config_path,
        config=config, selection_root=Path(args.selection_root).resolve(),
        reference_artifact=Path(args.reference_artifact).resolve(),
    )
    test_path = feature_cache / "test.pt"
    metadata_path = feature_cache / "metadata.json"
    if test_path.is_file():
        _, test, metadata = base._validate_train_cache(
            feature_cache, protocol, key, require_test=True
        )
        reference = read_reference_artifact(
            config, Path(args.reference_artifact).resolve()
        )["results"][key]
        expected_sha = reference["source_feature_metadata"].get("test_sha256")
        if metadata.get("authorization_id") != authorization["authorization_id"]:
            if expected_sha is None or _sha256_file(test_path) != expected_sha:
                raise RuntimeError("existing test cache is not bound to this or the P2B run")
        print(f"TEST CACHE RESTORED {key} shape={tuple(test['features'].shape)}", flush=True)
        return {"status": "restored", "test_sha256": _sha256_file(test_path)}
    base._validate_train_cache(feature_cache, protocol, key, require_test=False)
    dataset, backbone = config["datasets"][key], protocol["backbone"]
    base.random_initialization(config["selection"]["split_seed"])
    namespace = argparse.Namespace(
        dataset=dataset["dataset"], root=args.root,
        num_classes=dataset["num_classes"], num_tasks=dataset["num_tasks"],
        batch_size=args.batch_size, data_augmentation=backbone["preprocessing"],
        num_workers=args.num_workers,
    )
    _, loaders = base.load_dataset(namespace)
    ordered_loaders = _ordered_task_loaders(loaders, dataset["num_tasks"])
    device = torch.device(args.device)
    model = base.load_model(
        backbone["model_name"], checkpoint_path=args.backbone_checkpoint,
        expected_checkpoint_size=backbone["checkpoint_size"],
        expected_checkpoint_sha256=backbone["checkpoint_sha256"],
    ).eval().to(device)
    features, labels = [], []
    for task, loader in enumerate(ordered_loaders, 1):
        values, targets = base.feature_extract(model, loader, device)
        features.append(values.cpu()); labels.append(targets.cpu())
        print(f"TEST EXTRACT {key} task={task}/{dataset['num_tasks']} samples={len(targets)}", flush=True)
    packed = {"features": torch.cat(features), "labels": torch.cat(labels)}
    if (
        tuple(packed["features"].shape) != (dataset["test_samples"], backbone["feature_dim"])
        or tuple(packed["labels"].shape) != (dataset["test_samples"],)
        or not bool(torch.isfinite(packed["features"]).all())
    ):
        raise ValueError("held-out extraction tensor contract mismatch")
    base._atomic_torch(test_path, packed)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "test_shape": list(packed["features"].shape),
        "test_labels_shape": list(packed["labels"].shape),
        "test_features_materialized": True,
        "authorization_id": authorization["authorization_id"],
        "test_sha256": _sha256_file(test_path),
    })
    base._atomic_json(metadata_path, metadata)
    print(f"TEST CACHE COMPLETE {key} shape={tuple(packed['features'].shape)}", flush=True)
    return {"status": "complete", "test_sha256": metadata["test_sha256"]}


def _run_test_unit(path: Path, context_sha: str, label: str, evaluator) -> dict:
    restored = _load_unit(path, context_sha)
    if restored is not None:
        print(f"RESTORED {label}", flush=True)
        return restored
    print(f"START {label}", flush=True)
    result = evaluator()
    result = _save_unit(path, context_sha, result)
    print(f"DONE {label} status={result['status']}", flush=True)
    return result


def evaluate_dataset(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    protocol = _base_protocol(config)
    key = args.dataset_key
    selection_root = Path(args.selection_root).resolve()
    selections = _validate_selections(config_path, config, selection_root)
    authorization = _validate_authorization(
        path=Path(args.authorization).resolve(), config_path=config_path,
        config=config, selection_root=selection_root,
        reference_artifact=Path(args.reference_artifact).resolve(),
    )
    reference = read_reference_artifact(
        config, Path(args.reference_artifact).resolve()
    )["results"][key]
    dataset = {
        **config["datasets"][key],
        "matched_expand_dim": selections[key]["width"],
        "fly_ridge_lambda": selections[key]["ridge_lambda"],
    }
    audit = base._validate_dataset_audit(
        None if args.dataset_audit is None else Path(args.dataset_audit).resolve(),
        key, dataset,
    )
    feature_cache = Path(args.feature_cache_dir).resolve()
    train, test, metadata = base._validate_train_cache(
        feature_cache, protocol, key, require_test=True
    )
    expected_test_sha = reference["source_feature_metadata"].get("test_sha256")
    if (
        metadata.get("authorization_id") != authorization["authorization_id"]
        and (expected_test_sha is None or _sha256_file(feature_cache / "test.pt") != expected_test_sha)
    ):
        raise ValueError("test feature cache is not bound to the locked confirmation")
    stream = {
        "features": torch.cat((train["features"], test["features"])),
        "labels": torch.cat((train["labels"], test["labels"])),
    }
    offset = len(train["labels"])
    source_sha = _sha256_bytes((
        _sha256_file(feature_cache / "train.pt")
        + _sha256_file(feature_cache / "test.pt")
    ).encode())
    device = torch.device(args.device)
    output = Path(args.output_root).resolve() / key
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for replicate_index, replicate in enumerate(config["final_evaluation"]["replicates"]):
        reference_row = reference["seed_results"][replicate_index]
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(dataset["num_classes"]), dataset["num_classes"]
        )
        if reference_row["class_order"] != class_order:
            raise ValueError(f"reference class order mismatch: {key}/{replicate_index}")
        per_task = dataset["num_classes"] // dataset["num_tasks"]
        training_parts, test_parts = [], []
        for task in range(dataset["num_tasks"]):
            ids = torch.tensor(class_order[task * per_task:(task + 1) * per_task])
            training_parts.append(torch.nonzero(torch.isin(train["labels"], ids)).flatten())
            test_parts.append(torch.nonzero(torch.isin(test["labels"], ids)).flatten() + offset)
        cache = _prepare_code_cache(
            train=stream, train_sha256=source_sha,
            cache_dir=Path(args.code_cache_root).resolve() / key / f"replicate_{replicate_index}",
            config=_cache_config(config, key, replicate["projection_seed"]),
            device=args.device,
        )
        context = {
            "config_sha256": _sha256_file(config_path),
            "authorization_id": authorization["authorization_id"],
            "reference_artifact_sha256": config["p2b_reference"]["artifact_sha256"],
            "selection_sha256": selections[key]["sha256"],
            "dataset_key": key, "replicate": replicate,
            "class_order": class_order,
            "training_indices_sha256": _sequence_sha256(training_parts),
            "test_indices_sha256": _sequence_sha256(test_parts),
            "code_identity": cache[2]["identity_sha256"],
            "projection_sha256": _tensor_content_sha256(cache[3]),
            "source_identity": _source_identity(),
        }
        result = _run_test_unit(
            _unit_path(output, f"replicate_{replicate_index}_state_matched"),
            _sha256_bytes(json.dumps(context, sort_keys=True).encode()),
            f"CONFIRM {key} replicate={replicate_index} state-matched",
            lambda cache=cache, training_parts=training_parts,
            test_parts=test_parts, replicate=replicate:
                _evaluate_exact_matched(
                    manifest=base._runtime_manifest(protocol), dataset=dataset,
                    seed=replicate["projection_seed"], stream=stream,
                    code_indices=cache[0], code_values=cache[1], projection=cache[3],
                    training_parts=training_parts, test_parts=test_parts, device=device,
                ),
        )
        expected_state = exact_fly_state_bytes(
            feature_dim=config["state_matching"]["feature_dim"],
            expand_dim=selections[key]["width"],
            synaptic_degree=config["state_matching"]["synaptic_degree"],
            num_classes=dataset["num_classes"],
        )
        if result.get("status") != "complete":
            raise RuntimeError(f"state-matched method incomplete: {key}/{replicate_index}")
        if result["persistent_state_bytes"] != expected_state:
            raise AssertionError(
                f"runtime state mismatch: {result['persistent_state_bytes']} != {expected_state}"
            )
        rows.append({
            "replicate_index": replicate_index,
            "class_order_seed": replicate["class_order_seed"],
            "projection_seed": replicate["projection_seed"],
            "class_order": class_order,
            "method": result,
        })
        print(f"REPLICATE COMPLETE dataset={key} index={replicate_index}", flush=True)
    payload = {
        "schema_version": 1, "study_id": config["study_id"],
        "dataset_key": key, "status": "STATE_MATCHED_CONFIRMATION_COMPLETE",
        "uses_test_set": True, "test_tuning_allowed": False,
        "accuracy_based_early_stop": False,
        "prior_test_use_disclosure": config["p2b_reference"]["prior_test_use_disclosure"],
        "authorization_id": authorization["authorization_id"],
        "reference_artifact_sha256": config["p2b_reference"]["artifact_sha256"],
        "selection_sha256": selections[key]["sha256"],
        "selected_hyperparameters": {
            "width": selections[key]["width"],
            "ridge_lambda": selections[key]["ridge_lambda"],
        },
        "state_match": selections[key]["state_match"],
        "source_feature_metadata": metadata, "dataset_audit": audit,
        "legacy_processed_split_disclosure": (
            "19 cross-split duplicate hashes, including 18 conflicting-label "
            "hashes; not content-disjoint" if key == "imagenetr" else None
        ),
        "seed_results": rows,
    }
    base._atomic_json(output / "state_matched_results.json", payload)
    print(f"DATASET STATE-MATCHED CONFIRMATION COMPLETE {key}", flush=True)
    return payload


def _method_summary(results: list[dict]) -> dict:
    metrics = {}
    for metric in (
        "final_accuracy", "average_incremental_accuracy", "forgetting",
        "persistent_state_bytes", "total_update_seconds", "total_inference_seconds",
    ):
        metrics[metric] = _mean_std_ci([float(item[metric]) for item in results])
    return {"status": "complete", **metrics}


def summarize(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    reference = read_reference_artifact(
        config, Path(args.reference_artifact).resolve()
    )
    output = Path(args.output_root).resolve()
    summaries, paired_state, rows, curves = {}, {}, [], []
    for key in DATASET_KEYS:
        path = output / key / "state_matched_results.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing state-matched result: {key}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "STATE_MATCHED_CONFIRMATION_COMPLETE"
            or payload.get("uses_test_set") is not True
            or payload.get("test_tuning_allowed") is not False
            or payload.get("reference_artifact_sha256")
            != config["p2b_reference"]["artifact_sha256"]
        ):
            raise ValueError(f"invalid state-matched result contract: {key}")
        matched_results = [item["method"] for item in payload["seed_results"]]
        reference_rows = reference["results"][key]["seed_results"]
        summaries[key] = {
            name: reference["summary"]["dataset_method_summaries"][key][name]
            for name in REFERENCE_METHODS
        }
        summaries[key]["exact_fly_state_matched"] = _method_summary(matched_results)
        differences = []
        for replicate_index, (matched, reference_row) in enumerate(zip(matched_results, reference_rows)):
            p2b = reference_row["methods"]["srq_fly_p2b_10000"]
            if (
                payload["seed_results"][replicate_index]["class_order_seed"]
                != reference_row["class_order_seed"]
                or payload["seed_results"][replicate_index]["projection_seed"]
                != reference_row["projection_seed"]
            ):
                raise ValueError(f"paired replicate mismatch: {key}/{replicate_index}")
            differences.append(
                p2b["average_incremental_accuracy"]
                - matched["average_incremental_accuracy"]
            )
        paired_state[key] = _mean_std_ci(differences)
        for method, method_summary in summaries[key].items():
            row = {"dataset": key, "method": method, "status": "complete"}
            for metric, values in method_summary.items():
                if metric == "status":
                    continue
                for field in ("mean", "sample_std", "ci95_low", "ci95_high"):
                    row[f"{metric}_{field}"] = values[field]
            rows.append(row)
        for replicate_index, matched in enumerate(matched_results):
            for task, accuracy in enumerate(matched["stage_accuracy"], 1):
                curves.append({
                    "dataset": key, "method": "exact_fly_state_matched",
                    "replicate_index": replicate_index, "task": task,
                    "task_fraction": task / len(matched["stage_accuracy"]),
                    "average_seen_accuracy": accuracy,
                })
    summary = {
        "schema_version": 1, "study_id": config["study_id"],
        "status": "STATE_MATCHED_CONFIRMATION_REPORTED_WITHOUT_ACCURACY_GATE",
        "uses_test_set": True, "test_tuning_allowed": False,
        "prior_test_use_disclosure": config["p2b_reference"]["prior_test_use_disclosure"],
        "config_sha256": _sha256_file(config_path),
        "source_identity": _source_identity(),
        "reference_artifact_sha256": config["p2b_reference"]["artifact_sha256"],
        "dataset_method_summaries": summaries,
        "paired_p2b_minus_state_matched_fly_aia": paired_state,
        "paired_p2b_minus_same_width_exact_fly_aia": reference["summary"][
            "paired_p2b_minus_exact_fly_aia"
        ],
        "imagenetr_disclosure": (
            "legacy processed split with 19 cross-split duplicate hashes; "
            "not content-disjoint"
        ),
    }
    base._atomic_json(output / "state_matched_final_summary.json", summary)
    for path, data in (
        (output / "state_matched_metrics.csv", rows),
        (output / "state_matched_task_curves.csv", curves),
    ):
        fields = sorted({field for row in data for field in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--config", required=True)
    select.add_argument("--dataset-key", choices=DATASET_KEYS, required=True)
    select.add_argument("--feature-cache-dir", required=True)
    select.add_argument("--code-cache-root", required=True)
    select.add_argument("--output-root", required=True)
    select.add_argument("--dataset-audit")
    select.add_argument("--device", default="cpu")
    lock = sub.add_parser("lock")
    lock.add_argument("--config", required=True)
    lock.add_argument("--selection-root", required=True)
    lock.add_argument("--reference-artifact", required=True)
    lock.add_argument("--output-root", required=True)
    lock.add_argument("--require-clean-git", action="store_true")
    extract = sub.add_parser("extract-test")
    extract.add_argument("--config", required=True)
    extract.add_argument("--dataset-key", choices=DATASET_KEYS, required=True)
    extract.add_argument("--selection-root", required=True)
    extract.add_argument("--reference-artifact", required=True)
    extract.add_argument("--authorization", required=True)
    extract.add_argument("--feature-cache-dir", required=True)
    extract.add_argument("--root", required=True)
    extract.add_argument("--backbone-checkpoint", required=True)
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--batch-size", type=int, default=128)
    extract.add_argument("--num-workers", type=int, default=2)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--dataset-key", choices=DATASET_KEYS, required=True)
    evaluate.add_argument("--selection-root", required=True)
    evaluate.add_argument("--reference-artifact", required=True)
    evaluate.add_argument("--authorization", required=True)
    evaluate.add_argument("--feature-cache-dir", required=True)
    evaluate.add_argument("--code-cache-root", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--dataset-audit")
    evaluate.add_argument("--device", default="cpu")
    report = sub.add_parser("summarize")
    report.add_argument("--config", required=True)
    report.add_argument("--reference-artifact", required=True)
    report.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "select":
        select_dataset(args)
    elif args.command == "lock":
        lock_confirmation(args)
    elif args.command == "extract-test":
        extract_test(args)
    elif args.command == "evaluate":
        evaluate_dataset(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
