"""Locked three-dataset confirmation for the selected SRQ-FLY P2B backend.

The train-only ridge choices and final replicate identities are inherited from
the immutable self-contained study.  This runner never selects a
hyperparameter.  It compares Exact FLY, the optimized P2B implementation and
raw-feature Ridge on the same cached features, class order, projection and WTA
codes.  The involved test splits have been consumed before, so every output is
explicitly labelled as a backend confirmation rather than a fresh held-out
result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.srq_fly_optimized import SquareRootFLYLearner
from tools import srq_fly_selfcontained as base
from tools.srq_fly_heldout import (
    _assert_sample_free_inventory,
    _dense_codes,
    _evaluate_raw,
    _expand_cross,
    _inventory,
    _mean_std_ci,
    _result_metrics,
    _solve,
    _targets,
    _task_predictions,
)
from tools.tail_fly_phasea import _load_unit, _save_unit, _unit_path
from tools.twa_fly_pilot import (
    _sequence_sha256,
    _sha256_bytes,
    _sha256_file,
    _tensor_content_sha256,
)


DATASET_KEYS = ("cifar100", "cub200", "imagenetr")
METHODS = ("exact_fly_10000", "srq_fly_p2b_10000", "raw_ridge")


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


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version", "study_id", "base_protocol", "selection_evidence",
        "method_identity", "p2b_backend", "final_evaluation",
    }
    if set(config) != expected or config.get("schema_version") != 1:
        raise ValueError("P2B confirmation config schema mismatch")
    if set(config["selection_evidence"]["files"]) != set(DATASET_KEYS):
        raise ValueError("selection evidence must cover all three datasets")
    backend = config["p2b_backend"]
    if backend != {
        "storage_mode": "int8",
        "update_backend": "blocked_qr",
        "update_panel_size": 128,
        "first_update_backend": "gram_cholesky",
        "quantization_backend": "streaming",
        "quantization_batch_blocks": 64,
    }:
        raise ValueError("P2B backend contract changed")
    final = config["final_evaluation"]
    if (
        final.get("methods") != list(METHODS)
        or final.get("inherit_replicates_from_base_protocol") is not True
        or final.get("test_tuning_allowed") is not False
        or final.get("accuracy_based_early_stop") is not False
        or not final.get("prior_test_use_disclosure")
    ):
        raise ValueError("final confirmation contract mismatch")
    observed_identity = _source_identity().copy()
    observed_identity.pop("runner_sha256")
    if observed_identity != config["method_identity"]:
        raise ValueError("P2B confirmation source identity mismatch")
    return config


def _read_base_protocol(config: dict, path: Path) -> dict:
    expected = config["base_protocol"]
    if _sha256_file(path) != expected["sha256"]:
        raise ValueError("base protocol identity mismatch")
    runner = ROOT / expected["runner_path"]
    if _sha256_file(runner) != expected["runner_sha256"]:
        raise ValueError("base selection runner identity mismatch")
    protocol = base._read_protocol(path)
    base._verify_method_identity(protocol)
    if protocol["final_evaluation"]["methods"] != [
        "exact_fly_10000", "srq_fly_10000", "raw_ridge"
    ]:
        raise ValueError("base method identities changed")
    if len(protocol["final_evaluation"]["replicates"]) != 6:
        raise ValueError("base protocol no longer contains six final replicates")
    return protocol


def _validate_selections(
    config: dict, base_protocol_path: Path, selection_root: Path
) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    protocol_sha = config["base_protocol"]["sha256"]
    runner_sha = config["base_protocol"]["runner_sha256"]
    for key in DATASET_KEYS:
        path = selection_root / key / "selection.json"
        evidence = config["selection_evidence"]["files"][key]
        if not path.is_file() or _sha256_file(path) != evidence["sha256"]:
            raise ValueError(f"immutable selection evidence mismatch for {key}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "SELECTION_COMPLETE"
            or payload.get("uses_test_set") is not False
            or payload.get("held_out_test_authorized") is not False
            or payload.get("protocol_sha256") != protocol_sha
            or payload.get("runner_sha256") != runner_sha
            or float(payload.get("selected_fly_family_lambda"))
            != float(evidence["fly_ridge_lambda"])
            or float(payload.get("selected_raw_ridge_lambda"))
            != float(evidence["raw_ridge_lambda"])
        ):
            raise ValueError(f"selection contents mismatch for {key}")
        selected[key] = {
            "path": str(path),
            "sha256": evidence["sha256"],
            "fly_ridge_lambda": float(evidence["fly_ridge_lambda"]),
            "raw_ridge_lambda": float(evidence["raw_ridge_lambda"]),
        }
    if _sha256_file(base_protocol_path) != protocol_sha:
        raise ValueError("selection/base protocol binding mismatch")
    return selected


def lock_confirmation(
    *, config_path: Path, base_protocol_path: Path, selection_root: Path,
    base_authorization_path: Path, output_root: Path, require_clean_git: bool,
) -> dict:
    config = _read_config(config_path)
    _read_base_protocol(config, base_protocol_path)
    selections = _validate_selections(config, base_protocol_path, selection_root)
    base_authorization = base._validate_authorization(
        base_authorization_path, base_protocol_path, selection_root
    )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if require_clean_git and dirty:
        raise RuntimeError(f"confirmation lock requires a clean Git worktree:\n{dirty}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    record = {
        "schema_version": 1, "study_id": config["study_id"],
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": _sha256_file(config_path),
        "source_identity": _source_identity(),
        "base_protocol_sha256": config["base_protocol"]["sha256"],
        "base_authorization_id": base_authorization["authorization_id"],
        "selection_sha256": {
            key: selections[key]["sha256"] for key in DATASET_KEYS
        },
        "selected_hyperparameters": {
            key: {
                "fly_ridge_lambda": selections[key]["fly_ridge_lambda"],
                "raw_ridge_lambda": selections[key]["raw_ridge_lambda"],
            }
            for key in DATASET_KEYS
        },
        "p2b_backend": config["p2b_backend"],
        "git_commit": commit, "git_dirty": bool(dirty),
        "test_tuning_allowed": False, "accuracy_based_early_stop": False,
        "prior_test_use_disclosure": config["final_evaluation"][
            "prior_test_use_disclosure"
        ],
    }
    record["confirmation_authorization_id"] = _sha256_bytes(
        json.dumps(record, sort_keys=True).encode()
    )
    path = output_root / "confirmation_authorization.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        immutable = (
            "config_sha256", "source_identity", "base_protocol_sha256",
            "base_authorization_id", "selection_sha256",
            "selected_hyperparameters", "p2b_backend", "git_commit",
        )
        if any(previous.get(key) != record.get(key) for key in immutable):
            raise RuntimeError("existing confirmation authorization context mismatch")
        print(
            "P2B CONFIRMATION AUTHORIZATION RESTORED "
            f"id={previous['confirmation_authorization_id']}", flush=True,
        )
        return previous
    base._atomic_json(path, record)
    print(
        "P2B CONFIRMATION AUTHORIZATION LOCKED "
        f"id={record['confirmation_authorization_id']}", flush=True,
    )
    return record


def _validate_confirmation_authorization(
    *, path: Path, config_path: Path, config: dict,
    base_authorization: dict, selections: dict[str, dict],
) -> dict:
    if not path.is_file():
        raise FileNotFoundError("confirmation_authorization.json is missing")
    record = json.loads(path.read_text(encoding="utf-8"))
    claimed_id = record.get("confirmation_authorization_id")
    identity_payload = dict(record)
    identity_payload.pop("confirmation_authorization_id", None)
    recomputed_id = _sha256_bytes(
        json.dumps(identity_payload, sort_keys=True).encode()
    )
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    current_dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    expected_selections = {
        key: selections[key]["sha256"] for key in DATASET_KEYS
    }
    if (
        record.get("config_sha256") != _sha256_file(config_path)
        or record.get("source_identity") != _source_identity()
        or record.get("base_authorization_id")
        != base_authorization["authorization_id"]
        or record.get("selection_sha256") != expected_selections
        or record.get("p2b_backend") != config["p2b_backend"]
        or claimed_id != recomputed_id
        or record.get("git_commit") != current_commit
        or record.get("git_dirty") is not False
        or bool(current_dirty)
        or record.get("test_tuning_allowed") is not False
        or record.get("accuracy_based_early_stop") is not False
    ):
        raise ValueError("P2B confirmation authorization mismatch")
    return record


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _p2b_learner(
    *, config: dict, protocol: dict, ridge_lambda: float, projection: torch.Tensor,
    seed: int, device: torch.device,
) -> SquareRootFLYLearner:
    representation = protocol["representation"]
    backend = config["p2b_backend"]
    return SquareRootFLYLearner(
        feature_dim=protocol["backbone"]["feature_dim"],
        expand_dim=representation["expand_dim"],
        synaptic_degree=representation["synaptic_degree"],
        coding_level=representation["coding_level"],
        ridge_lambda=ridge_lambda,
        block_size=representation["block_size"],
        group_size=representation["group_size"],
        seed=seed,
        device=device,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        projection=projection,
        **backend,
    )


def _evaluate_paired_p2b(
    *, config: dict, manifest: dict, dataset: dict, seed: int, stream: dict,
    code_indices: torch.Tensor, code_values: torch.Tensor,
    projection: torch.Tensor, training_parts: list[torch.Tensor],
    test_parts: list[torch.Tensor], device: torch.device,
) -> dict:
    common = manifest["representation"]
    dimension = common["large_expand_dim"]
    dtype = torch.float32
    gram = torch.zeros((dimension, dimension), device=device, dtype=dtype)
    cross = torch.zeros((dimension, 0), device=device, dtype=dtype)
    counts = torch.zeros(0, device=device, dtype=dtype)
    class_ids: list[int] = []
    learner = _p2b_learner(
        config=config, protocol={
            "backbone": manifest["backbone"],
            "representation": {
                "expand_dim": dimension,
                "synaptic_degree": common["synaptic_degree"],
                "coding_level": common["coding_level"],
                "block_size": common["block_size"],
                "group_size": common["group_size"],
            },
        }, ridge_lambda=dataset["fly_ridge_lambda"], projection=projection,
        seed=seed, device=device,
    )
    exact_matrix: list[list[float]] = []
    p2b_matrix: list[list[float]] = []
    exact_residuals: list[float] = []
    p2b_residuals: list[float] = []
    agreements: list[float] = []
    logit_errors: list[float] = []
    exact_state_by_task: list[int] = []
    p2b_state_by_task: list[int] = []
    timing: list[dict] = []
    exact_weights = None
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        code_started = time.perf_counter()
        codes = _dense_codes(
            code_indices[indices], code_values[indices], dimension, device, dtype
        )
        _sync(device)
        code_seconds = time.perf_counter() - code_started
        labels = stream["labels"][indices]

        exact_started = time.perf_counter()
        new_ids = sorted(set(class_ids) | set(map(int, labels.tolist())))
        cross, counts = _expand_cross(cross, counts, class_ids, new_ids)
        class_ids = new_ids
        targets = _targets(labels, class_ids, device=device, dtype=dtype)
        gram += codes.T @ codes
        cross += codes.T @ targets
        counts += targets.sum(0)
        system = gram + dataset["fly_ridge_lambda"] * torch.eye(
            dimension, device=device, dtype=dtype
        )
        exact_weights, exact_residual = _solve(system, cross)
        _sync(device)
        exact_seconds = time.perf_counter() - exact_started

        p2b_started = time.perf_counter()
        learner.update_codes_consuming(codes, labels)
        _sync(device)
        p2b_seconds = time.perf_counter() - p2b_started
        learner.assert_exemplar_free_state()
        if learner.class_ids != class_ids:
            raise AssertionError("Exact FLY and P2B class columns diverged")
        del codes, system

        exact_inference_started = time.perf_counter()
        exact_row, exact_predictions, exact_logits = _task_predictions(
            weights=exact_weights, class_ids=class_ids, parts=test_parts, task=task,
            code_indices=code_indices, code_values=code_values,
            labels=stream["labels"], dimension=dimension,
            batch_size=common["evaluation_batch_size"],
        )
        _sync(device)
        exact_inference_seconds = time.perf_counter() - exact_inference_started
        p2b_inference_started = time.perf_counter()
        p2b_row, p2b_predictions, p2b_logits = _task_predictions(
            weights=learner.weights, class_ids=class_ids, parts=test_parts, task=task,
            code_indices=code_indices, code_values=code_values,
            labels=stream["labels"], dimension=dimension,
            batch_size=common["evaluation_batch_size"],
        )
        _sync(device)
        p2b_inference_seconds = time.perf_counter() - p2b_inference_started

        agreements.append(statistics.fmean([
            float((left == right).float().mean())
            for left, right in zip(exact_predictions, p2b_predictions)
        ]))
        numerator = sum(
            float(((right - left) ** 2).sum())
            for left, right in zip(exact_logits, p2b_logits)
        )
        denominator = max(
            sum(float((left ** 2).sum()) for left in exact_logits), 1.0
        )
        logit_errors.append(math.sqrt(numerator / denominator))
        exact_matrix.append(exact_row)
        p2b_matrix.append(p2b_row)
        exact_residuals.append(exact_residual)
        p2b_residuals.append(float(learner.diagnostics["solver_relative_residual"]))

        exact_inventory = _inventory({
            "projection": projection, "G": gram, "Q": cross,
            "counts": counts, "weights": exact_weights,
        })
        p2b_inventory = _inventory(learner.persistent_tensors())
        structural = {
            dimension, manifest["backbone"]["feature_dim"], len(class_ids)
        }
        _assert_sample_free_inventory(exact_inventory, learner.total_rows, structural)
        _assert_sample_free_inventory(p2b_inventory, learner.total_rows, structural)
        exact_state_by_task.append(sum(item["bytes"] for item in exact_inventory))
        p2b_state_by_task.append(sum(item["bytes"] for item in p2b_inventory))
        timing.append({
            "task": task + 1,
            "shared_code_materialization_seconds": code_seconds,
            "exact_update_seconds": exact_seconds,
            "p2b_update_seconds": p2b_seconds,
            "exact_inference_seconds": exact_inference_seconds,
            "p2b_inference_seconds": p2b_inference_seconds,
        })
        print(
            f"TASK paired-p2b seed={seed} {task+1}/{len(training_parts)} "
            f"exact={statistics.fmean(exact_row):.4f} "
            f"p2b={statistics.fmean(p2b_row):.4f} "
            f"agree={100.0 * agreements[-1]:.3f}%",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    exact_metrics = _result_metrics(exact_matrix)
    p2b_metrics = _result_metrics(p2b_matrix)
    base_fields = {
        "status": "complete", "uses_test_set": True, "exemplar_free": True,
        "ridge_lambda": dataset["fly_ridge_lambda"],
    }
    wall_seconds = time.perf_counter() - started
    return {
        "status": "complete", "uses_test_set": True,
        "exact": {
            **base_fields, "method": "exact_fly_10000", **exact_metrics,
            "accuracy_matrix": exact_matrix,
            "persistent_state_bytes": exact_state_by_task[-1],
            "persistent_state_bytes_by_task": exact_state_by_task,
            "persistent_tensor_inventory": exact_inventory,
            "maximum_solver_relative_residual": max(exact_residuals),
            "total_update_seconds": sum(x["exact_update_seconds"] for x in timing),
            "total_inference_seconds": sum(
                x["exact_inference_seconds"] for x in timing
            ),
            "timing": timing, "seconds": wall_seconds,
        },
        "p2b": {
            **base_fields, "method": "srq_fly_p2b_10000", **p2b_metrics,
            "accuracy_matrix": p2b_matrix,
            "persistent_state_bytes": p2b_state_by_task[-1],
            "persistent_state_bytes_by_task": p2b_state_by_task,
            "persistent_tensor_inventory": p2b_inventory,
            "maximum_solver_relative_residual": max(p2b_residuals),
            "minimum_prediction_agreement": min(agreements),
            "maximum_relative_logit_frobenius_error": max(logit_errors),
            "total_update_seconds": sum(x["p2b_update_seconds"] for x in timing),
            "total_inference_seconds": sum(
                x["p2b_inference_seconds"] for x in timing
            ),
            "timing": timing, "seconds": wall_seconds,
        },
        "prediction_agreement_by_task": agreements,
        "relative_logit_frobenius_error_by_task": logit_errors,
        "timing": timing,
    }


def _run_unit(path: Path, context_sha: str, label: str, evaluator) -> dict:
    result = _load_unit(path, context_sha)
    if result is not None:
        print(f"RESTORED {label}", flush=True)
        return result
    print(f"START {label}", flush=True)
    result = evaluator()
    result = _save_unit(path, context_sha, result)
    print(f"DONE {label} status={result['status']}", flush=True)
    return result


def evaluate_dataset(args) -> dict:
    config_path = Path(args.config).resolve()
    base_protocol_path = Path(args.base_protocol).resolve()
    selection_root = Path(args.selection_root).resolve()
    config = _read_config(config_path)
    protocol = _read_base_protocol(config, base_protocol_path)
    selections = _validate_selections(config, base_protocol_path, selection_root)
    authorization = base._validate_authorization(
        Path(args.authorization).resolve(), base_protocol_path, selection_root
    )
    confirmation_authorization = _validate_confirmation_authorization(
        path=Path(args.confirmation_authorization).resolve(),
        config_path=config_path, config=config,
        base_authorization=authorization, selections=selections,
    )
    key = args.dataset_key
    dataset_static = protocol["datasets"][key]
    audit = base._validate_dataset_audit(
        None if args.dataset_audit is None else Path(args.dataset_audit).resolve(),
        key, dataset_static,
    )
    feature_cache = Path(args.feature_cache_dir).resolve()
    train, test, metadata = base._validate_train_cache(
        feature_cache, protocol, key, require_test=True
    )
    if metadata.get("authorization_id") != authorization["authorization_id"]:
        raise ValueError("test cache authorization mismatch")
    selected = selections[key]
    dataset = {
        **dataset_static,
        "fly_ridge_lambda": selected["fly_ridge_lambda"],
        "raw_ridge_lambda": selected["raw_ridge_lambda"],
    }
    manifest = base._runtime_manifest(protocol)
    stream = {
        "features": torch.cat((train["features"], test["features"])),
        "labels": torch.cat((train["labels"], test["labels"])),
    }
    offset = len(train["labels"])
    device = torch.device(args.device)
    source_sha = _sha256_bytes((
        _sha256_file(feature_cache / "train.pt")
        + _sha256_file(feature_cache / "test.pt")
    ).encode())
    output = Path(args.output_root).resolve() / key
    output.mkdir(parents=True, exist_ok=True)
    code_root = Path(args.code_cache_root).resolve()
    seed_results = []
    identity = _source_identity()
    for replicate_index, replicate in enumerate(
        protocol["final_evaluation"]["replicates"]
    ):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(dataset["num_classes"]), dataset["num_classes"]
        )
        classes_per_task = dataset["num_classes"] // dataset["num_tasks"]
        training_parts, test_parts = [], []
        for task in range(dataset["num_tasks"]):
            ids = torch.tensor(class_order[
                task * classes_per_task:(task + 1) * classes_per_task
            ])
            training_parts.append(torch.nonzero(torch.isin(train["labels"], ids)).flatten())
            test_parts.append(
                torch.nonzero(torch.isin(test["labels"], ids)).flatten() + offset
            )
        cache = base._prepare_code_cache(
            train=stream,
            train_sha256=source_sha,
            cache_dir=code_root / key / f"replicate_{replicate_index}",
            config=base._cache_config(
                protocol, dataset, replicate["projection_seed"]
            ),
            device=args.device,
        )
        context = {
            "config_sha256": _sha256_file(config_path),
            "base_protocol_sha256": config["base_protocol"]["sha256"],
            "authorization_id": authorization["authorization_id"],
            "confirmation_authorization_id": confirmation_authorization[
                "confirmation_authorization_id"
            ],
            "selection_sha256": selected["sha256"],
            "dataset_key": key, "replicate": replicate,
            "class_order": class_order,
            "train_indices_sha256": _sequence_sha256(training_parts),
            "test_indices_sha256": _sequence_sha256(test_parts),
            "code_identity": cache[2]["identity_sha256"],
            "projection_sha256": _tensor_content_sha256(cache[3]),
            "source_identity": identity,
        }
        context_sha = _sha256_bytes(json.dumps(context, sort_keys=True).encode())
        paired = _run_unit(
            _unit_path(output, f"replicate_{replicate_index}_paired_p2b"),
            context_sha,
            f"CONFIRM {key} replicate={replicate_index} exact+p2b",
            lambda cache=cache, training_parts=training_parts,
            test_parts=test_parts, replicate=replicate: _evaluate_paired_p2b(
                config=config, manifest=manifest, dataset=dataset,
                seed=replicate["projection_seed"], stream=stream,
                code_indices=cache[0], code_values=cache[1], projection=cache[3],
                training_parts=training_parts, test_parts=test_parts, device=device,
            ),
        )
        raw = _run_unit(
            _unit_path(output, f"replicate_{replicate_index}_raw"),
            context_sha,
            f"CONFIRM {key} replicate={replicate_index} raw",
            lambda training_parts=training_parts, test_parts=test_parts,
            replicate=replicate: _evaluate_raw(
                manifest=manifest, dataset=dataset,
                seed=replicate["class_order_seed"], stream=stream,
                training_parts=training_parts, test_parts=test_parts, device=device,
            ),
        )
        methods = {
            "exact_fly_10000": paired["exact"],
            "srq_fly_p2b_10000": paired["p2b"],
            "raw_ridge": raw,
        }
        seed_results.append({
            "replicate_index": replicate_index,
            "class_order_seed": replicate["class_order_seed"],
            "projection_seed": replicate["projection_seed"],
            "class_order": class_order, "methods": methods,
            "paired_diagnostics": {
                "prediction_agreement_by_task": paired[
                    "prediction_agreement_by_task"
                ],
                "relative_logit_frobenius_error_by_task": paired[
                    "relative_logit_frobenius_error_by_task"
                ],
            },
        })
        print(f"REPLICATE COMPLETE dataset={key} index={replicate_index}", flush=True)
    payload = {
        "schema_version": 1, "study_id": config["study_id"],
        "dataset_key": key, "status": "CONFIRMATION_COMPLETE",
        "uses_test_set": True, "test_tuning_allowed": False,
        "accuracy_based_early_stop": False,
        "prior_test_use_disclosure": config["final_evaluation"][
            "prior_test_use_disclosure"
        ],
        "authorization_id": authorization["authorization_id"],
        "confirmation_authorization_id": confirmation_authorization[
            "confirmation_authorization_id"
        ],
        "selected_hyperparameters": {
            "fly_family_lambda": dataset["fly_ridge_lambda"],
            "raw_ridge_lambda": dataset["raw_ridge_lambda"],
        },
        "p2b_backend": config["p2b_backend"],
        "source_identity": identity,
        "source_feature_metadata": metadata, "dataset_audit": audit,
        "legacy_processed_split_disclosure": (
            "19 cross-split duplicate hashes, including 18 conflicting-label "
            "hashes; not content-disjoint" if key == "imagenetr" else None
        ),
        "seed_results": seed_results,
    }
    base._atomic_json(output / "confirmation_results.json", payload)
    print(f"DATASET CONFIRMATION COMPLETE {key}", flush=True)
    return payload


def summarize(config_path: Path, output_root: Path) -> dict:
    config = _read_config(config_path)
    rows: list[dict] = []
    curves: list[dict] = []
    paired: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    for key in DATASET_KEYS:
        path = output_root / key / "confirmation_results.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing confirmation result: {key}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "CONFIRMATION_COMPLETE"
            or payload.get("uses_test_set") is not True
            or payload.get("test_tuning_allowed") is not False
        ):
            raise ValueError(f"invalid confirmation result contract: {key}")
        summaries[key] = {}
        for method in METHODS:
            results = [x["methods"][method] for x in payload["seed_results"]]
            if any(x.get("status") != "complete" for x in results):
                raise ValueError(f"incomplete method {key}/{method}")
            metrics = {}
            for metric in (
                "final_accuracy", "average_incremental_accuracy", "forgetting",
                "persistent_state_bytes", "total_update_seconds",
                "total_inference_seconds",
            ):
                metrics[metric] = _mean_std_ci([
                    float(x[metric]) for x in results
                ])
            summaries[key][method] = {"status": "complete", **metrics}
            row = {"dataset": key, "method": method, "status": "complete"}
            for metric, values in metrics.items():
                for field in ("mean", "sample_std", "ci95_low", "ci95_high"):
                    row[f"{metric}_{field}"] = values[field]
            rows.append(row)
            for replicate_index, result in enumerate(results):
                for task, value in enumerate(result["stage_accuracy"], 1):
                    curves.append({
                        "dataset": key, "method": method,
                        "replicate_index": replicate_index, "task": task,
                        "task_fraction": task / len(result["stage_accuracy"]),
                        "average_seen_accuracy": value,
                    })
        differences = [
            x["methods"]["srq_fly_p2b_10000"]["average_incremental_accuracy"]
            - x["methods"]["exact_fly_10000"]["average_incremental_accuracy"]
            for x in payload["seed_results"]
        ]
        paired[key] = _mean_std_ci(differences)
    summary = {
        "schema_version": 1, "study_id": config["study_id"],
        "status": "CONFIRMATION_REPORTED_WITHOUT_ACCURACY_GATE",
        "uses_test_set": True, "test_tuning_allowed": False,
        "prior_test_use_disclosure": config["final_evaluation"][
            "prior_test_use_disclosure"
        ],
        "config_sha256": _sha256_file(config_path),
        "source_identity": _source_identity(),
        "dataset_method_summaries": summaries,
        "paired_p2b_minus_exact_fly_aia": paired,
        "imagenetr_disclosure": (
            "legacy processed split with 19 cross-split duplicate hashes; "
            "not content-disjoint"
        ),
    }
    base._atomic_json(output_root / "final_confirmation_summary.json", summary)
    for path, data in (
        (output_root / "metrics_summary.csv", rows),
        (output_root / "task_curves.csv", curves),
    ):
        fields = sorted({field for row in data for field in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock")
    lock.add_argument("--config", required=True)
    lock.add_argument("--base-protocol", required=True)
    lock.add_argument("--selection-root", required=True)
    lock.add_argument("--authorization", required=True)
    lock.add_argument("--output-root", required=True)
    lock.add_argument("--require-clean-git", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--base-protocol", required=True)
    evaluate.add_argument("--dataset-key", choices=DATASET_KEYS, required=True)
    evaluate.add_argument("--selection-root", required=True)
    evaluate.add_argument("--authorization", required=True)
    evaluate.add_argument("--confirmation-authorization", required=True)
    evaluate.add_argument("--feature-cache-dir", required=True)
    evaluate.add_argument("--code-cache-root", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--dataset-audit")
    evaluate.add_argument("--device", default="cpu")
    report = sub.add_parser("summarize")
    report.add_argument("--config", required=True)
    report.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "lock":
        lock_confirmation(
            config_path=Path(args.config).resolve(),
            base_protocol_path=Path(args.base_protocol).resolve(),
            selection_root=Path(args.selection_root).resolve(),
            base_authorization_path=Path(args.authorization).resolve(),
            output_root=Path(args.output_root).resolve(),
            require_clean_git=args.require_clean_git,
        )
    elif args.command == "evaluate":
        evaluate_dataset(args)
    else:
        summarize(Path(args.config).resolve(), Path(args.output_root).resolve())


if __name__ == "__main__":
    main()
