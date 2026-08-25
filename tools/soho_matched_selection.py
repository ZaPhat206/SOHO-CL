"""Matched train-only selection for SOHO, official FLY, tuned FLY and raw Ridge.

This is a separate V2 protocol.  It deliberately imports the locked V1 runner
for shared data, metric and learner-fidelity code while adding an independent
FLY representation search.  V1 source files and authorizations remain intact.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from tools import soho_selfcontained as base
from models.backbone import load_model
from utils.data_utils import load_dataset
from utils.train_utils import feature_extract, random_initialization


ROOT = Path(__file__).resolve().parents[1]
DATASET_KEYS = base.DATASET_KEYS
METHODS = (
    "soho_replay_fidelity",
    "flycl_fidelity",
    "flycl_validation_tuned",
    "raw_ridge",
)


def _read_protocol(path: Path) -> dict:
    protocol = base._read_protocol(path)
    if base._sha256_file(ROOT / "tools" / "soho_selfcontained.py") != protocol.get("base_runner_sha256"):
        raise ValueError("locked V1 base-runner identity mismatch")
    matched = protocol.get("matched_fly_selection", {})
    if (
        tuple(matched.get("synaptic_degree_grid", [])) != (100, 300, 500)
        or tuple(matched.get("coding_level_grid", [])) != (0.1, 0.2, 0.3, 0.4, 0.45, 0.5)
        or matched.get("near_tie_tolerance_pp") != 0.05
        or tuple(protocol.get("matched_final_methods", [])) != METHODS
    ):
        raise ValueError("matched FLY selection contract mismatch")
    return protocol


def _fly_candidates(protocol: dict) -> list[dict]:
    search = protocol["matched_fly_selection"]
    return [
        {"synaptic_degree": int(degree), "coding_level": float(coding)}
        for degree in search["synaptic_degree_grid"]
        for coding in search["coding_level_grid"]
    ]


def _protocol_with_fly(protocol: dict, candidate: dict) -> dict:
    configured = copy.deepcopy(protocol)
    configured["fly_fixed"]["synaptic_degree"] = int(candidate["synaptic_degree"])
    configured["fly_fixed"]["coding_level"] = float(candidate["coding_level"])
    return configured


def _select_fly_near_tie(results: list[dict], tolerance_pp: float) -> tuple[dict, float]:
    valid = [item for item in results if item.get("valid")]
    if not valid:
        raise RuntimeError("no valid tuned-FLY candidate")
    best = max(float(item["mean_inner_aia"]) for item in valid)
    eligible = [item for item in valid if best - float(item["mean_inner_aia"]) <= tolerance_pp]
    selected = min(
        eligible,
        key=lambda item: (
            int(item["config"]["synaptic_degree"]),
            float(item["config"]["coding_level"]),
        ),
    )
    return selected, best


def _source(protocol_path: Path, feature_cache_dir: Path) -> dict:
    return {
        "protocol_sha256": base._sha256_file(protocol_path),
        "runner_sha256": base._sha256_file(Path(__file__).resolve()),
        "base_runner_sha256": base._sha256_file(ROOT / "tools" / "soho_selfcontained.py"),
        "train_sha256": base._sha256_file(feature_cache_dir / "train.pt"),
    }


def _ordered_test_loaders(test_loaders) -> list:
    """Return loaders in task order without ever iterating mapping keys as loaders."""
    if isinstance(test_loaders, dict):
        return [test_loaders[task] for task in sorted(test_loaders)]
    return list(test_loaders)


def select_dataset(*, protocol_path: Path, dataset_key: str, feature_cache_dir: Path,
                   output_root: Path, dataset_audit_path: Path | None,
                   device_name: str) -> dict:
    protocol = _read_protocol(protocol_path)
    base_payload = base.select_dataset(
        protocol_path=protocol_path,
        dataset_key=dataset_key,
        feature_cache_dir=feature_cache_dir,
        output_root=output_root,
        dataset_audit_path=dataset_audit_path,
        device_name=device_name,
    )
    dataset = protocol["datasets"][dataset_key]
    train, _, _ = base._validate_cache(
        feature_cache_dir, protocol, dataset_key, require_test=False
    )
    output_dir = output_root / dataset_key
    source = {
        **_source(protocol_path, feature_cache_dir),
        "method_identity": base._verify_method_identity(protocol),
    }
    replicates = []
    selection = protocol["selection"]
    for index, replicate in enumerate(selection["development_replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(dataset["num_classes"]), dataset["num_classes"]
        )
        parts = base._nested_parts(
            train["labels"], class_order, dataset["num_tasks"], selection["split_seed"],
            selection["outer_validation_fraction"], selection["inner_validation_fraction"],
        )
        replicates.append({
            "index": index,
            "replicate": replicate,
            "class_order": class_order,
            "parts": parts,
        })

    fly_results = []
    for candidate_index, candidate in enumerate(_fly_candidates(protocol)):
        candidate_protocol = _protocol_with_fly(protocol, candidate)
        per_replicate = []
        for item in replicates:
            inner_fit, inner_validation = item["parts"][0], item["parts"][1]
            context = {
                **source,
                "phase": "inner_fly_candidate",
                "dataset_key": dataset_key,
                "candidate": candidate,
                "replicate": item["replicate"],
                "class_order": item["class_order"],
            }
            result = base._unit(
                output_dir / "matched_units" / f"inner_fly_c{candidate_index}_r{item['index']}.json",
                context,
                lambda candidate_protocol=candidate_protocol, item=item,
                       inner_fit=inner_fit, inner_validation=inner_validation:
                    base._evaluate(
                        "flycl_fidelity", candidate_protocol, dataset,
                        item["replicate"]["projection_seed"], train,
                        inner_fit, inner_validation,
                        base_payload["selected_soho_config"],
                        float(base_payload["selected_raw_ridge_lambda"]),
                        device_name, False,
                    ),
            )
            per_replicate.append(result)
        valid = all(item.get("status") == "complete" for item in per_replicate)
        fly_results.append({
            "candidate_index": candidate_index,
            "config": candidate,
            "valid": valid,
            "mean_inner_aia": statistics.fmean(
                item["average_incremental_accuracy"] for item in per_replicate
            ) if valid else None,
            "per_replicate": per_replicate,
        })

    selected_fly, best_fly_score = _select_fly_near_tie(
        fly_results, float(protocol["matched_fly_selection"]["near_tie_tolerance_pp"])
    )
    selected_protocol = _protocol_with_fly(protocol, selected_fly["config"])
    for item, confirmation in zip(replicates, base_payload["outer_confirmation"]):
        outer_fit, outer_validation = item["parts"][2], item["parts"][3]
        context = {
            **source,
            "phase": "outer_confirmation",
            "dataset_key": dataset_key,
            "method": "flycl_validation_tuned",
            "replicate": item["replicate"],
            "class_order": item["class_order"],
            "fly_config": selected_fly["config"],
        }
        result = base._unit(
            output_dir / "matched_units" / f"outer_flycl_validation_tuned_r{item['index']}.json",
            context,
            lambda selected_protocol=selected_protocol, item=item,
                   outer_fit=outer_fit, outer_validation=outer_validation:
                base._evaluate(
                    "flycl_fidelity", selected_protocol, dataset,
                    item["replicate"]["projection_seed"], train,
                    outer_fit, outer_validation,
                    base_payload["selected_soho_config"],
                    float(base_payload["selected_raw_ridge_lambda"]),
                    device_name, False,
                ),
        )
        result["method"] = "flycl_validation_tuned"
        confirmation["methods"]["flycl_validation_tuned"] = result

    base_payload.update(source)
    base_payload.update({
        "study_id": protocol["study_id"],
        "matched_selection": True,
        "fly_official_config": {
            "synaptic_degree": int(protocol["fly_fixed"]["synaptic_degree"]),
            "coding_level": float(protocol["fly_fixed"]["coding_level"]),
        },
        "fly_search_space": _fly_candidates(protocol),
        "fly_candidates": fly_results,
        "selected_fly_config": selected_fly["config"],
        "selected_fly_mean_inner_aia": selected_fly["mean_inner_aia"],
        "best_fly_mean_inner_aia": best_fly_score,
        "matched_method_labels": list(METHODS),
        "fairness_disclosure": (
            "SOHO, tuned FLY and raw Ridge select method-specific hyperparameters on identical "
            "inner splits and development replicates; official FLY remains an untuned fidelity control"
        ),
    })
    base._atomic_json(output_dir / "selection.json", base_payload)
    print(
        f"MATCHED SELECTION COMPLETE dataset={dataset_key} "
        f"fly={selected_fly['config']} mean_inner_AIA={selected_fly['mean_inner_aia']:.4f}",
        flush=True,
    )
    return base_payload


def lock_selection(protocol_path: Path, selection_root: Path, output_root: Path,
                   require_clean_git: bool) -> dict:
    protocol = _read_protocol(protocol_path)
    selections, selected = {}, {}
    for key in DATASET_KEYS:
        path = selection_root / key / "selection.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing matched selection: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "SELECTION_COMPLETE"
            or payload.get("uses_test_set") is not False
            or payload.get("held_out_test_authorized") is not False
            or payload.get("protocol_sha256") != base._sha256_file(protocol_path)
            or payload.get("runner_sha256") != base._sha256_file(Path(__file__).resolve())
            or payload.get("base_runner_sha256") != protocol["base_runner_sha256"]
            or payload.get("selected_fly_config") not in _fly_candidates(protocol)
        ):
            raise ValueError(f"matched selection contract mismatch for {key}")
        selections[key] = {"path": str(path), "sha256": base._sha256_file(path)}
        selected[key] = {
            "soho_config": payload["selected_soho_config"],
            "fly_official_config": payload["fly_official_config"],
            "fly_validation_tuned_config": payload["selected_fly_config"],
            "raw_ridge_lambda": float(payload["selected_raw_ridge_lambda"]),
        }
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    git_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip())
    if require_clean_git and git_dirty:
        raise RuntimeError("final lock requires a clean Git worktree")
    record = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": base._sha256_file(protocol_path),
        "runner_sha256": base._sha256_file(Path(__file__).resolve()),
        "base_runner_sha256": protocol["base_runner_sha256"],
        "method_identity": base._verify_method_identity(protocol),
        "selection_files": selections,
        "selected_hyperparameters": selected,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "test_tuning_allowed": False,
        "accuracy_based_early_stop": False,
    }
    record["authorization_id"] = base._sha256_json(record)
    path = output_root / "authorization.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        immutable = (
            "protocol_sha256", "runner_sha256", "base_runner_sha256", "method_identity",
            "selection_files", "selected_hyperparameters", "git_commit",
        )
        if any(previous.get(field) != record.get(field) for field in immutable):
            raise RuntimeError("existing authorization belongs to different code/selection")
        print(f"AUTHORIZATION RESTORED id={previous['authorization_id']}", flush=True)
        return previous
    base._atomic_json(path, record)
    print(f"AUTHORIZATION LOCKED id={record['authorization_id']}", flush=True)
    return record


def _validate_authorization(path: Path, protocol_path: Path,
                            selection_root: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError("authorization.json is missing")
    protocol = _read_protocol(protocol_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("protocol_sha256") != base._sha256_file(protocol_path)
        or record.get("runner_sha256") != base._sha256_file(Path(__file__).resolve())
        or record.get("base_runner_sha256") != protocol["base_runner_sha256"]
        or record.get("test_tuning_allowed") is not False
    ):
        raise ValueError("authorization source identity mismatch")
    for key in DATASET_KEYS:
        if record["selection_files"][key]["sha256"] != base._sha256_file(
            selection_root / key / "selection.json"
        ):
            raise ValueError(f"selection changed after authorization: {key}")
    return record


def extract_test(*, protocol_path: Path, dataset_key: str, authorization_path: Path,
                 selection_root: Path, feature_cache_dir: Path, dataset_root: str,
                 checkpoint_path: str, device_name: str, batch_size: int,
                 num_workers: int) -> dict:
    protocol = _read_protocol(protocol_path)
    authorization = _validate_authorization(
        authorization_path, protocol_path, selection_root
    )
    dataset, backbone = protocol["datasets"][dataset_key], protocol["backbone"]
    test_path, metadata_path = feature_cache_dir / "test.pt", feature_cache_dir / "metadata.json"
    if test_path.is_file():
        _, test, metadata = base._validate_cache(
            feature_cache_dir, protocol, dataset_key, require_test=True
        )
        if metadata.get("authorization_id") != authorization["authorization_id"]:
            raise RuntimeError("existing test cache belongs to a different authorization")
        print(f"TEST CACHE RESTORED {dataset_key} shape={tuple(test['features'].shape)}", flush=True)
        return {"status": "restored", "test_sha256": base._sha256_file(test_path)}
    base._validate_cache(feature_cache_dir, protocol, dataset_key, require_test=False)
    random_initialization(protocol["selection"]["split_seed"])
    namespace = argparse.Namespace(
        dataset=dataset["dataset"], root=dataset_root,
        num_classes=dataset["num_classes"], num_tasks=dataset["num_tasks"],
        batch_size=batch_size, data_augmentation=backbone["preprocessing"],
        num_workers=num_workers,
    )
    _, test_loaders = load_dataset(namespace)
    device = torch.device(device_name)
    model = load_model(
        backbone["model_name"], checkpoint_path=checkpoint_path,
        expected_checkpoint_size=backbone["checkpoint_size"],
        expected_checkpoint_sha256=backbone["checkpoint_sha256"],
    ).eval().to(device)
    features, labels = [], []
    for task, loader in enumerate(_ordered_test_loaders(test_loaders)):
        values, targets = feature_extract(model, loader, device)
        features.append(values.cpu())
        labels.append(targets.cpu())
        print(
            f"TEST EXTRACT {dataset_key} task={task+1}/{dataset['num_tasks']} "
            f"samples={len(targets)}",
            flush=True,
        )
    packed = {"features": torch.cat(features), "labels": torch.cat(labels)}
    if (
        tuple(packed["features"].shape) != (dataset["test_samples"], backbone["feature_dim"])
        or tuple(packed["labels"].shape) != (dataset["test_samples"],)
        or not bool(torch.isfinite(packed["features"]).all())
    ):
        raise ValueError("test extraction tensor contract mismatch")
    base._atomic_torch(test_path, packed)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "test_shape": list(packed["features"].shape),
        "test_labels_shape": list(packed["labels"].shape),
        "test_features_materialized": True,
        "authorization_id": authorization["authorization_id"],
        "test_sha256": base._sha256_file(test_path),
    })
    base._atomic_json(metadata_path, metadata)
    print(f"TEST CACHE COMPLETE {dataset_key} shape={tuple(packed['features'].shape)}", flush=True)
    return {"status": "complete", "test_sha256": metadata["test_sha256"]}


def _evaluate_method(method: str, protocol: dict, selected_fly: dict, dataset: dict,
                     seed: int, stream: dict, train_parts: list[torch.Tensor],
                     test_parts: list[torch.Tensor], soho_config: dict,
                     raw_ridge: float, device_name: str) -> dict:
    actual_method, configured = method, protocol
    if method == "flycl_validation_tuned":
        actual_method = "flycl_fidelity"
        configured = _protocol_with_fly(protocol, selected_fly)
    result = base._evaluate(
        actual_method, configured, dataset, seed, stream, train_parts, test_parts,
        soho_config, raw_ridge, device_name, True,
    )
    result["method"] = method
    return result


def evaluate_dataset(*, protocol_path: Path, dataset_key: str,
                     selection_root: Path, authorization_path: Path,
                     feature_cache_dir: Path, output_root: Path,
                     dataset_audit_path: Path | None, device_name: str) -> dict:
    protocol = _read_protocol(protocol_path)
    base._verify_method_identity(protocol)
    authorization = _validate_authorization(
        authorization_path, protocol_path, selection_root
    )
    dataset = protocol["datasets"][dataset_key]
    audit = base._validate_dataset_audit(dataset_audit_path, dataset_key, dataset)
    train, test, metadata = base._validate_cache(
        feature_cache_dir, protocol, dataset_key, require_test=True
    )
    selection_path = selection_root / dataset_key / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    soho_config = selection["selected_soho_config"]
    selected_fly = selection["selected_fly_config"]
    raw_ridge = float(selection["selected_raw_ridge_lambda"])
    stream = {
        "features": torch.cat((train["features"], test["features"])),
        "labels": torch.cat((train["labels"], test["labels"])),
    }
    test_offset = len(train["labels"])
    output_dir = output_root / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "protocol_sha256": base._sha256_file(protocol_path),
        "runner_sha256": base._sha256_file(Path(__file__).resolve()),
        "base_runner_sha256": protocol["base_runner_sha256"],
        "authorization_id": authorization["authorization_id"],
        "selection_sha256": base._sha256_file(selection_path),
        "train_sha256": base._sha256_file(feature_cache_dir / "train.pt"),
        "test_sha256": base._sha256_file(feature_cache_dir / "test.pt"),
    }
    seed_results = []
    for replicate_index, replicate in enumerate(protocol["final_evaluation"]["replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(dataset["num_classes"]), dataset["num_classes"]
        )
        train_parts = base._task_parts(
            train["labels"], class_order, dataset["num_tasks"]
        )
        test_parts = base._task_parts(
            test["labels"], class_order, dataset["num_tasks"], test_offset
        )
        methods = {}
        for method in METHODS:
            context = {
                **source,
                "dataset_key": dataset_key,
                "replicate": replicate,
                "class_order": class_order,
                "method": method,
                "soho_config": soho_config,
                "selected_fly_config": selected_fly,
                "raw_ridge": raw_ridge,
            }
            methods[method] = base._unit(
                output_dir / "matched_units" / f"final_r{replicate_index}_{method}.json",
                context,
                lambda method=method, replicate=replicate,
                       train_parts=train_parts, test_parts=test_parts:
                    _evaluate_method(
                        method, protocol, selected_fly, dataset,
                        replicate["projection_seed"], stream, train_parts, test_parts,
                        soho_config, raw_ridge, device_name,
                    ),
            )
        seed_results.append({
            "replicate_index": replicate_index,
            "class_order_seed": replicate["class_order_seed"],
            "projection_seed": replicate["projection_seed"],
            "class_order": class_order,
            "methods": methods,
        })
        print(f"REPLICATE COMPLETE dataset={dataset_key} index={replicate_index}", flush=True)
    failures = [
        {"replicate_index": item["replicate_index"], "method": method,
         "status": result.get("status")}
        for item in seed_results for method, result in item["methods"].items()
        if result.get("status") != "complete"
    ]
    payload = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "dataset_key": dataset_key,
        "status": "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
        "uses_test_set": True,
        "test_tuning_allowed": False,
        "accuracy_based_early_stop": False,
        "authorization_id": authorization["authorization_id"],
        "selected_hyperparameters": {
            "soho_config": soho_config,
            "fly_official_config": selection["fly_official_config"],
            "fly_validation_tuned_config": selected_fly,
            "raw_ridge_lambda": raw_ridge,
        },
        "fairness_disclosure": selection["fairness_disclosure"],
        "source_feature_metadata": metadata,
        "dataset_audit": audit,
        "environment": base._environment(device_name),
        "feature_cache_disk_bytes": sum(
            path.stat().st_size for path in feature_cache_dir.iterdir() if path.is_file()
        ),
        "legacy_processed_split_disclosure": (
            "19 cross-split duplicate hashes including 18 conflicting-label hashes; not content-disjoint"
            if dataset_key == "imagenetr" else None
        ),
        "soho_replay_disclosure": (
            "SOHO retains historical frozen-backbone features and labels; state bytes include both"
        ),
        "seed_results": seed_results,
        "failures": failures,
    }
    base._atomic_json(output_dir / "final_results.json", payload)
    print(f"DATASET COMPLETE {dataset_key} status={payload['status']}", flush=True)
    return payload


def summarize(protocol_path: Path, output_root: Path) -> dict:
    protocol = _read_protocol(protocol_path)
    rows, curves, paired, summaries = [], [], {}, {}
    for key in DATASET_KEYS:
        path = output_root / key / "final_results.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing final result: {key}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("uses_test_set") is not True or payload.get("test_tuning_allowed") is not False:
            raise ValueError(f"invalid final-result contract: {key}")
        summaries[key] = {}
        for method in METHODS:
            results = [item["methods"][method] for item in payload["seed_results"]]
            if any(result.get("status") != "complete" for result in results):
                summaries[key][method] = {"status": "incomplete"}
                rows.append({"dataset": key, "method": method, "status": "incomplete"})
                continue
            metrics = {
                metric: base._mean_std_ci([float(result[metric]) for result in results])
                for metric in (
                    "final_accuracy", "average_incremental_accuracy", "forgetting",
                    "persistent_state_bytes", "total_update_seconds",
                    "total_inference_seconds", "peak_runtime_memory_bytes",
                )
            }
            summaries[key][method] = {"status": "complete", **metrics}
            row = {"dataset": key, "method": method, "status": "complete"}
            for metric, values in metrics.items():
                for field, value in values.items():
                    row[f"{metric}_{field}"] = value
            rows.append(row)
            for replicate_index, result in enumerate(results):
                for task, value in enumerate(result["stage_accuracy"], 1):
                    curves.append({
                        "dataset": key,
                        "method": method,
                        "replicate_index": replicate_index,
                        "task": task,
                        "task_fraction": task / len(result["stage_accuracy"]),
                        "average_seen_accuracy": value,
                    })
        comparisons = (
            ("soho_replay_fidelity", "flycl_fidelity"),
            ("soho_replay_fidelity", "flycl_validation_tuned"),
            ("flycl_validation_tuned", "flycl_fidelity"),
            ("soho_replay_fidelity", "raw_ridge"),
        )
        for left, right in comparisons:
            differences = [
                item["methods"][left]["average_incremental_accuracy"]
                - item["methods"][right]["average_incremental_accuracy"]
                for item in payload["seed_results"]
                if item["methods"][left].get("status") == "complete"
                and item["methods"][right].get("status") == "complete"
            ]
            paired.setdefault(key, {})[f"{left}_minus_{right}"] = base._mean_std_ci(differences)
    summary = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "status": "REPORTED_WITHOUT_ACCURACY_GATE",
        "uses_test_set": True,
        "dataset_method_summaries": summaries,
        "paired_aia_differences": paired,
        "soho_exemplar_free": False,
        "fairness_disclosure": (
            "SOHO, tuned FLY and raw Ridge use method-specific train-only selection; "
            "official FLY is retained as a separate fidelity control"
        ),
        "test_reuse_disclosure": (
            "Repository test splits were used by earlier phases; this is a locked comparative "
            "evaluation, not a first-use untouched held-out study."
        ),
        "imagenetr_disclosure": (
            "legacy processed split with 19 duplicate hashes; not content-disjoint"
        ),
    }
    base._atomic_json(output_root / "final_summary.json", summary)
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
    select = sub.add_parser("select")
    select.add_argument("--protocol", required=True)
    select.add_argument("--dataset-key", choices=DATASET_KEYS, required=True)
    select.add_argument("--feature-cache-dir", required=True)
    select.add_argument("--output-root", required=True)
    select.add_argument("--dataset-audit")
    select.add_argument("--device", default="cpu")
    lock = sub.add_parser("lock")
    lock.add_argument("--protocol", required=True)
    lock.add_argument("--selection-root", required=True)
    lock.add_argument("--output-root", required=True)
    lock.add_argument("--require-clean-git", action="store_true")
    extract = sub.add_parser("extract-test")
    extract.add_argument("--protocol", required=True)
    extract.add_argument("--dataset-key", choices=DATASET_KEYS, required=True)
    extract.add_argument("--selection-root", required=True)
    extract.add_argument("--authorization", required=True)
    extract.add_argument("--feature-cache-dir", required=True)
    extract.add_argument("--root", required=True)
    extract.add_argument("--backbone-checkpoint", required=True)
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--batch-size", type=int, default=128)
    extract.add_argument("--num-workers", type=int, default=2)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--dataset-key", choices=DATASET_KEYS, required=True)
    evaluate.add_argument("--selection-root", required=True)
    evaluate.add_argument("--authorization", required=True)
    evaluate.add_argument("--feature-cache-dir", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--dataset-audit")
    evaluate.add_argument("--device", default="cpu")
    report = sub.add_parser("summarize")
    report.add_argument("--protocol", required=True)
    report.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    protocol_path = Path(args.protocol).resolve()
    if args.command == "select":
        select_dataset(
            protocol_path=protocol_path,
            dataset_key=args.dataset_key,
            feature_cache_dir=Path(args.feature_cache_dir).resolve(),
            output_root=Path(args.output_root).resolve(),
            dataset_audit_path=None if args.dataset_audit is None else Path(args.dataset_audit).resolve(),
            device_name=args.device,
        )
    elif args.command == "lock":
        lock_selection(
            protocol_path, Path(args.selection_root).resolve(),
            Path(args.output_root).resolve(), args.require_clean_git,
        )
    elif args.command == "extract-test":
        extract_test(
            protocol_path=protocol_path,
            dataset_key=args.dataset_key,
            authorization_path=Path(args.authorization).resolve(),
            selection_root=Path(args.selection_root).resolve(),
            feature_cache_dir=Path(args.feature_cache_dir).resolve(),
            dataset_root=args.root,
            checkpoint_path=args.backbone_checkpoint,
            device_name=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.command == "evaluate":
        evaluate_dataset(
            protocol_path=protocol_path,
            dataset_key=args.dataset_key,
            selection_root=Path(args.selection_root).resolve(),
            authorization_path=Path(args.authorization).resolve(),
            feature_cache_dir=Path(args.feature_cache_dir).resolve(),
            output_root=Path(args.output_root).resolve(),
            dataset_audit_path=None if args.dataset_audit is None else Path(args.dataset_audit).resolve(),
            device_name=args.device,
        )
    else:
        summarize(protocol_path, Path(args.output_root).resolve())


if __name__ == "__main__":
    main()
