"""Locked train-only selection and final evaluation for repository SOHO.

This runner deliberately evaluates the current replay-based SOHO semantics via
``CachedSOHOReplayFidelity``.  It does not claim exemplar freedom: historical
backbone features and labels retained by SOHO are inventoried and counted as
persistent learner state.
"""

from __future__ import annotations

import argparse
import csv
import gc
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

from methods.cached_replay_baselines import CachedFlyCLFidelity, CachedSOHOReplayFidelity
from models.backbone import load_model
from utils.data_utils import load_dataset
from utils.train_utils import feature_extract, random_initialization


DATASET_KEYS = ("cifar100", "cub200", "imagenetr")
METHODS = ("soho_replay_fidelity", "flycl_fidelity", "raw_ridge")
T_CRITICAL_95_DF5 = 2.570581835636305
DENSITY_GRID = (0.1, 0.2, 0.3, 0.5, 0.8)
CODING_LEVEL_GRID = (0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.8)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _environment(device_name: str) -> dict:
    device = torch.device(device_name)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device_name,
        "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else None,
    }


def _read_protocol(path: str | Path) -> dict:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or tuple(protocol.get("datasets", {})) != DATASET_KEYS:
        raise ValueError("SOHO protocol schema mismatch")
    backbone = protocol["backbone"]
    if (
        backbone.get("model_name") != "vit_base_patch16_224"
        or backbone.get("feature_dim") != 768
        or backbone.get("checkpoint_sha256")
        != "32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b"
    ):
        raise ValueError("backbone contract mismatch")
    selection = protocol["selection"]
    if (
        selection.get("split_seed") != 2025
        or selection.get("outer_validation_fraction") != 0.2
        or selection.get("inner_validation_fraction") != 0.2
        or tuple(selection.get("density_grid", [])) != DENSITY_GRID
        or tuple(selection.get("coding_level_grid", [])) != CODING_LEVEL_GRID
        or selection.get("use_etf") is not True
        or selection.get("anchor_density") != 0.3
        or selection.get("anchor_coding_level") != 0.3
        or selection.get("top_density_count") != 2
        or selection.get("top_coding_count") != 3
        or selection.get("sparse_tie_tolerance_pp") != 0.05
        or len(selection.get("raw_ridge_grid", [])) != 12
        or len(selection.get("development_replicates", [])) != 3
    ):
        raise ValueError("train-only selection contract mismatch")
    final = protocol["final_evaluation"]
    if (
        final.get("methods") != list(METHODS)
        or len(final.get("replicates", [])) != 6
        or final.get("test_tuning_allowed") is not False
        or final.get("accuracy_based_early_stop") is not False
    ):
        raise ValueError("final-evaluation contract mismatch")
    development = {
        (item["class_order_seed"], item["projection_seed"])
        for item in selection["development_replicates"]
    }
    final_seeds = {
        (item["class_order_seed"], item["projection_seed"])
        for item in final["replicates"]
    }
    if len(development) != 3 or len(final_seeds) != 6 or development & final_seeds:
        raise ValueError("development and final replicate identities must be disjoint")
    return protocol


def _soho_candidates(protocol: dict) -> list[dict]:
    selection = protocol["selection"]
    return [
        {"density": float(density), "coding_level": float(coding), "use_etf": True}
        for density in selection["density_grid"]
        for coding in selection["coding_level_grid"]
    ]


def _rank_stage1_sensitivity(stage1_results: list[dict], selection: dict):
    anchor_density = float(selection["anchor_density"])
    anchor_coding = float(selection["anchor_coding_level"])
    density = [item for item in stage1_results if item["config"]["coding_level"] == anchor_coding]
    coding = [item for item in stage1_results if item["config"]["density"] == anchor_density]
    density = sorted(density, key=lambda item: (-item["mean_inner_aia"], item["config"]["density"]))
    coding = sorted(coding, key=lambda item: (-item["mean_inner_aia"], item["config"]["coding_level"]))
    return density, coding


def _select_sparse_near_tie(stage2_results: list[dict], tolerance_pp: float) -> tuple[dict, float]:
    valid = [item for item in stage2_results if item["valid"]]
    if not valid:
        raise RuntimeError("no valid SOHO stage-2 interaction candidate")
    best = max(item["mean_inner_aia"] for item in valid)
    near = [item for item in valid if item["mean_inner_aia"] >= best - tolerance_pp]
    selected = min(
        near,
        key=lambda item: (item["config"]["coding_level"], item["config"]["density"]),
    )
    return selected, best


def _verify_method_identity(protocol: dict) -> dict:
    observed = {
        "soho_model_sha256": _sha256_file(ROOT / "models/soho.py"),
        "sohocl_sha256": _sha256_file(ROOT / "methods/sohocl.py"),
        "cached_baselines_sha256": _sha256_file(ROOT / "methods/cached_replay_baselines.py"),
        "flyhash_sha256": _sha256_file(ROOT / "models/flyhash.py"),
    }
    if observed != protocol["method_identity"]:
        raise ValueError("SOHO/FLY method source identity mismatch")
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


def _validate_cache(cache_dir: str | Path, protocol: dict, key: str, require_test: bool):
    cache_dir = Path(cache_dir)
    dataset, backbone = protocol["datasets"][key], protocol["backbone"]
    metadata_path, train_path, test_path = (
        cache_dir / "metadata.json",
        cache_dir / "train.pt",
        cache_dir / "test.pt",
    )
    if not metadata_path.is_file() or not train_path.is_file():
        raise FileNotFoundError("feature cache requires metadata.json and train.pt")
    if require_test and not test_path.is_file():
        raise FileNotFoundError("final evaluation requires test.pt")
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
    for name, packed, expected_rows in (
        ("train", train, dataset["train_samples"]),
        ("test", test, dataset["test_samples"]),
    ):
        if packed is None:
            continue
        if (
            set(packed) != {"features", "labels"}
            or tuple(packed["features"].shape) != (expected_rows, backbone["feature_dim"])
            or tuple(packed["labels"].shape) != (expected_rows,)
            or not bool(torch.isfinite(packed["features"]).all())
            or sorted(map(int, torch.unique(packed["labels"]).tolist()))
            != list(range(dataset["num_classes"]))
        ):
            raise ValueError(f"invalid {name} feature cache")
    return train, test, metadata


def _nested_parts(labels: torch.Tensor, class_order: list[int], tasks: int, split_seed: int,
                  outer_fraction: float, inner_fraction: float):
    per_class = {}
    for class_id in sorted(map(int, torch.unique(labels).tolist())):
        indices = torch.nonzero(labels == class_id).flatten()
        generator = torch.Generator().manual_seed(split_seed * 1000 + class_id)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_outer = max(1, int(round(len(indices) * outer_fraction)))
        development, outer_validation = indices[n_outer:], indices[:n_outer]
        n_inner = max(1, int(round(len(development) * inner_fraction)))
        inner_fit, inner_validation = development[n_inner:], development[:n_inner]
        if min(map(len, (inner_fit, inner_validation, outer_validation))) <= 0:
            raise ValueError(f"empty nested partition for class {class_id}")
        per_class[class_id] = (inner_fit, inner_validation, development, outer_validation)
    classes_per_task = len(class_order) // tasks
    grouped = [[], [], [], []]
    for task in range(tasks):
        class_ids = class_order[task * classes_per_task:(task + 1) * classes_per_task]
        for part in range(4):
            grouped[part].append(torch.cat([per_class[class_id][part] for class_id in class_ids]))
    return tuple(grouped)


def _task_parts(labels: torch.Tensor, class_order: list[int], tasks: int, offset: int = 0):
    classes_per_task = len(class_order) // tasks
    parts = []
    for task in range(tasks):
        class_ids = torch.tensor(class_order[task * classes_per_task:(task + 1) * classes_per_task])
        parts.append(torch.nonzero(torch.isin(labels, class_ids)).flatten() + offset)
    return parts


def _forgetting(matrix: list[list[float]]) -> float:
    if len(matrix) <= 1:
        return 0.0
    values = []
    for task in range(len(matrix) - 1):
        best = max(matrix[stage][task] for stage in range(task, len(matrix) - 1))
        values.append(best - matrix[-1][task])
    return statistics.fmean(values)


def _metrics(matrix: list[list[float]]) -> dict:
    stage_accuracy = [statistics.fmean(row) for row in matrix]
    return {
        "accuracy_matrix": matrix,
        "stage_accuracy": stage_accuracy,
        "final_accuracy": stage_accuracy[-1],
        "average_incremental_accuracy": statistics.fmean(stage_accuracy),
        "forgetting": _forgetting(matrix),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _RawRidge:
    is_exemplar_free = True

    def __init__(self, feature_dim: int, num_classes: int, ridge: float, device: str):
        self.feature_dim, self.num_classes = feature_dim, num_classes
        self.ridge, self.device = float(ridge), torch.device(device)
        self.G = torch.zeros((feature_dim, feature_dim), dtype=torch.float64, device=self.device)
        self.Q = torch.zeros((feature_dim, num_classes), dtype=torch.float64, device=self.device)
        self.weights = None

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        x = features.to(self.device, torch.float64)
        y = torch.nn.functional.one_hot(labels.to(self.device), self.num_classes).to(torch.float64)
        self.G += x.T @ x
        self.Q += x.T @ y
        system = self.G + self.ridge * torch.eye(self.feature_dim, dtype=torch.float64, device=self.device)
        self.weights = torch.linalg.solve(system, self.Q)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("raw Ridge has not been updated")
        return (features.to(self.device, torch.float64) @ self.weights).argmax(1).cpu()

    def persistent_state_bytes(self) -> int:
        tensors = [self.G, self.Q] + ([] if self.weights is None else [self.weights])
        return sum(t.numel() * t.element_size() for t in tensors)


def _build_learner(method: str, protocol: dict, dataset: dict, seed: int, device: str,
                   soho_config: dict, raw_ridge: float):
    feature_dim = protocol["backbone"]["feature_dim"]
    fixed = protocol["soho_fixed"]
    if method == "soho_replay_fidelity":
        return CachedSOHOReplayFidelity(
            feature_dim, fixed["expand_dim"], soho_config["density"], fixed["olda_dim"],
            soho_config["use_etf"], soho_config["coding_level"], dataset["num_classes"],
            fixed["ridge_lower"], fixed["ridge_upper"], seed=seed, device=device,
            replay_chunk_size=fixed["replay_chunk_size"], gcv_sample_size=fixed["gcv_sample_size"],
        )
    if method == "flycl_fidelity":
        fly = protocol["fly_fixed"]
        return CachedFlyCLFidelity(
            feature_dim, fly["expand_dim"], fly["synaptic_degree"], fly["coding_level"],
            dataset["num_classes"], fly["ridge_lower"], fly["ridge_upper"],
            seed=seed, device=device,
        )
    if method == "raw_ridge":
        return _RawRidge(feature_dim, dataset["num_classes"], raw_ridge, device)
    raise ValueError(f"unsupported method: {method}")


def _state_audit(learner, method: str, retained_rows: int) -> dict:
    if method == "soho_replay_fidelity":
        feature_rows = sum(int(value.shape[0]) for value in learner.feature_history)
        label_rows = sum(int(value.shape[0]) for value in learner.label_history)
        if learner.is_exemplar_free or feature_rows != retained_rows or label_rows != retained_rows:
            raise AssertionError("SOHO replay-state contract mismatch")
        state = learner.state_dict()
        if "feature_history" not in state or "label_history" not in state:
            raise AssertionError("SOHO checkpoint failed to disclose sample-level replay")
        return {
            "exemplar_free": False,
            "historical_feature_rows": feature_rows,
            "historical_label_rows": label_rows,
            "replay_disclosure": "checkpoint retains all historical frozen-backbone features and labels",
        }
    if not bool(getattr(learner, "is_exemplar_free", True)):
        raise AssertionError(f"unexpected non-exemplar-free baseline: {method}")
    if method == "flycl_fidelity":
        forbidden = [name for name in learner.persistent_tensors() if "history" in name or "sample" in name]
        if forbidden:
            raise AssertionError(f"sample-level FLY state detected: {forbidden}")
    return {
        "exemplar_free": True,
        "historical_feature_rows": 0,
        "historical_label_rows": 0,
        "replay_disclosure": None,
    }


def _evaluate(method: str, protocol: dict, dataset: dict, seed: int, stream: dict,
              training_parts: list[torch.Tensor], evaluation_parts: list[torch.Tensor],
              soho_config: dict, raw_ridge: float, device_name: str, uses_test_set: bool) -> dict:
    device = torch.device(device_name)
    learner = _build_learner(method, protocol, dataset, seed, device_name, soho_config, raw_ridge)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    matrix, update_seconds, inference_seconds, selected_ridges = [], [], [], []
    retained_rows = 0
    try:
        for task, training_indices in enumerate(training_parts):
            _sync(device); started = time.perf_counter()
            learner.update(stream["features"][training_indices], stream["labels"][training_indices])
            _sync(device); update_seconds.append(time.perf_counter() - started)
            retained_rows += len(training_indices)
            audit = _state_audit(learner, method, retained_rows)
            selected_ridges.append(getattr(learner, "last_ridge", raw_ridge))
            row = []
            for previous in range(task + 1):
                indices = evaluation_parts[previous]
                _sync(device); started = time.perf_counter()
                predictions = learner.predict(stream["features"][indices])
                _sync(device); inference_seconds.append(time.perf_counter() - started)
                labels = stream["labels"][indices].cpu()
                row.append(float((predictions == labels).float().mean().item() * 100))
            matrix.append(row)
            print(
                f"TASK method={method} task={task+1}/{len(training_parts)} "
                f"seen_AA={statistics.fmean(row):.4f} state_MiB={learner.persistent_state_bytes()/2**20:.2f}",
                flush=True,
            )
        result = {
            "status": "complete",
            "uses_test_set": uses_test_set,
            "method": method,
            **_metrics(matrix),
            "persistent_state_bytes": learner.persistent_state_bytes(),
            "total_update_seconds": sum(update_seconds),
            "total_inference_seconds": sum(inference_seconds),
            "peak_runtime_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None,
            "selected_ridge_by_task": selected_ridges,
            "state_audit": audit,
        }
    finally:
        del learner
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def _unit(path: Path, context: dict, evaluator) -> dict:
    context_sha = _sha256_json(context)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("context_sha256") != context_sha:
            raise RuntimeError(f"resume context mismatch: {path}")
        print(f"RESTORED {path.stem}", flush=True)
        return payload["result"]
    print(f"START {path.stem}", flush=True)
    started = time.perf_counter()
    try:
        result = evaluator()
    except (RuntimeError, torch.linalg.LinAlgError) as error:
        result = {"status": "numerical_failure", "failure": f"{type(error).__name__}: {error}"}
    _atomic_json(path, {
        "context_sha256": context_sha,
        "unit_seconds": time.perf_counter() - started,
        "result": result,
    })
    print(f"DONE {path.stem} status={result['status']}", flush=True)
    return result


def select_dataset(*, protocol_path: Path, dataset_key: str, feature_cache_dir: Path,
                   output_root: Path, dataset_audit_path: Path | None, device_name: str) -> dict:
    protocol = _read_protocol(protocol_path)
    identities = _verify_method_identity(protocol)
    dataset = protocol["datasets"][dataset_key]
    audit = _validate_dataset_audit(dataset_audit_path, dataset_key, dataset)
    train, _, metadata = _validate_cache(feature_cache_dir, protocol, dataset_key, require_test=False)
    selection = protocol["selection"]
    candidates = _soho_candidates(protocol)
    raw_grid = list(map(float, selection["raw_ridge_grid"]))
    output_dir = output_root / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "protocol_sha256": _sha256_file(protocol_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "train_sha256": _sha256_file(feature_cache_dir / "train.pt"),
        "method_identity": identities,
    }
    replicates = []
    for index, replicate in enumerate(selection["development_replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(dataset["num_classes"]), dataset["num_classes"]
        )
        parts = _nested_parts(
            train["labels"], class_order, dataset["num_tasks"], selection["split_seed"],
            selection["outer_validation_fraction"], selection["inner_validation_fraction"],
        )
        replicates.append({"index": index, "replicate": replicate, "class_order": class_order, "parts": parts})

    evaluated_soho = {}

    def evaluate_soho_candidate(candidate: dict) -> dict:
        key = (float(candidate["density"]), float(candidate["coding_level"]))
        if key in evaluated_soho:
            return evaluated_soho[key]
        candidate_index = candidates.index(candidate)
        per_replicate = []
        for item in replicates:
            inner_fit, inner_validation = item["parts"][0], item["parts"][1]
            context = {
                **source, "phase": "inner_soho_candidate", "dataset_key": dataset_key,
                "candidate": candidate, "replicate": item["replicate"],
                "class_order": item["class_order"],
            }
            result = _unit(
                output_dir / "units" / f"inner_soho_c{candidate_index}_r{item['index']}.json",
                context,
                lambda candidate=candidate, item=item, inner_fit=inner_fit, inner_validation=inner_validation:
                    _evaluate(
                        "soho_replay_fidelity", protocol, dataset,
                        item["replicate"]["projection_seed"], train, inner_fit, inner_validation,
                        candidate, raw_grid[0], device_name, False,
                    ),
            )
            per_replicate.append(result)
        valid = all(result.get("status") == "complete" for result in per_replicate)
        summary = {
            "candidate_index": candidate_index,
            "config": candidate,
            "valid": valid,
            "mean_inner_aia": statistics.fmean(
                result["average_incremental_accuracy"] for result in per_replicate
            ) if valid else None,
            "per_replicate": per_replicate,
        }
        evaluated_soho[key] = summary
        return summary

    anchor_density = float(selection["anchor_density"])
    anchor_coding = float(selection["anchor_coding_level"])
    stage1_configs = [
        {"density": float(density), "coding_level": anchor_coding, "use_etf": True}
        for density in selection["density_grid"]
    ]
    stage1_configs.extend(
        {"density": anchor_density, "coding_level": float(coding), "use_etf": True}
        for coding in selection["coding_level_grid"] if float(coding) != anchor_coding
    )
    stage1_results = [evaluate_soho_candidate(candidate) for candidate in stage1_configs]
    if not all(item["valid"] for item in stage1_results):
        raise RuntimeError("invalid SOHO stage-1 sensitivity candidate")
    density_ranking, coding_ranking = _rank_stage1_sensitivity(stage1_results, selection)
    top_densities = [
        item["config"]["density"] for item in density_ranking[:selection["top_density_count"]]
    ]
    top_codings = [
        item["config"]["coding_level"] for item in coding_ranking[:selection["top_coding_count"]]
    ]
    stage2_configs = [
        {"density": density, "coding_level": coding, "use_etf": True}
        for density in top_densities for coding in top_codings
    ]
    stage2_results = [evaluate_soho_candidate(candidate) for candidate in stage2_configs]

    raw_results = []
    reference_soho = candidates[0]
    for ridge_index, ridge in enumerate(raw_grid):
        per_replicate = []
        for item in replicates:
            inner_fit, inner_validation = item["parts"][0], item["parts"][1]
            context = {
                **source, "phase": "inner_raw", "dataset_key": dataset_key,
                "ridge": ridge, "replicate": item["replicate"], "class_order": item["class_order"],
            }
            result = _unit(
                output_dir / "units" / f"inner_raw_l{ridge_index}_r{item['index']}.json",
                context,
                lambda ridge=ridge, item=item, inner_fit=inner_fit, inner_validation=inner_validation:
                    _evaluate(
                        "raw_ridge", protocol, dataset, item["replicate"]["projection_seed"],
                        train, inner_fit, inner_validation, reference_soho, ridge, device_name, False,
                    ),
            )
            per_replicate.append(result)
        valid = all(result.get("status") == "complete" for result in per_replicate)
        raw_results.append({
            "ridge_lambda": ridge,
            "valid": valid,
            "mean_inner_aia": statistics.fmean(
                result["average_incremental_accuracy"] for result in per_replicate
            ) if valid else None,
            "per_replicate": per_replicate,
        })

    valid_raw = [item for item in raw_results if item["valid"]]
    if not valid_raw:
        raise RuntimeError("no valid train-only candidate")
    selected_soho, best_soho_score = _select_sparse_near_tie(
        stage2_results, selection["sparse_tie_tolerance_pp"]
    )
    selected_raw = max(valid_raw, key=lambda item: (item["mean_inner_aia"], item["ridge_lambda"]))
    raw_boundary = selected_raw["ridge_lambda"] in {raw_grid[0], raw_grid[-1]}

    outer_confirmation = []
    for item in replicates:
        outer_fit, outer_validation = item["parts"][2], item["parts"][3]
        methods = {}
        settings = {
            "soho_replay_fidelity": (selected_soho["config"], selected_raw["ridge_lambda"]),
            "flycl_fidelity": (selected_soho["config"], selected_raw["ridge_lambda"]),
            "raw_ridge": (selected_soho["config"], selected_raw["ridge_lambda"]),
        }
        for method, (soho_config, raw_ridge) in settings.items():
            context = {
                **source, "phase": "outer_confirmation", "dataset_key": dataset_key,
                "method": method, "replicate": item["replicate"],
                "soho_config": soho_config, "raw_ridge": raw_ridge,
            }
            methods[method] = _unit(
                output_dir / "units" / f"outer_{method}_r{item['index']}.json",
                context,
                lambda method=method, soho_config=soho_config, raw_ridge=raw_ridge, item=item,
                       outer_fit=outer_fit, outer_validation=outer_validation:
                    _evaluate(
                        method, protocol, dataset, item["replicate"]["projection_seed"], train,
                        outer_fit, outer_validation, soho_config, raw_ridge, device_name, False,
                    ),
            )
        outer_confirmation.append({"replicate": item["replicate"], "methods": methods})

    payload = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "dataset_key": dataset_key,
        "status": "STOP_RAW_GRID_BOUNDARY" if raw_boundary else "SELECTION_COMPLETE",
        "uses_test_set": False,
        "held_out_test_authorized": False,
        **source,
        "selection_protocol": (
            "official train only; class-stratified 80/20 outer split and 80/20 inner split "
            "of the development partition; locked two-stage marginal sensitivity then local interaction"
        ),
        "soho_search_space": candidates,
        "selection_strategy": {
            "anchor_density": anchor_density,
            "anchor_coding_level": anchor_coding,
            "top_density_count": selection["top_density_count"],
            "top_coding_count": selection["top_coding_count"],
            "sparse_tie_tolerance_pp": selection["sparse_tie_tolerance_pp"],
            "tie_break": "within tolerance choose lower coding level, then lower density",
        },
        "raw_ridge_grid": raw_grid,
        "selected_soho_config": selected_soho["config"],
        "selected_soho_mean_inner_aia": selected_soho["mean_inner_aia"],
        "best_stage2_mean_inner_aia": best_soho_score,
        "selected_raw_ridge_lambda": selected_raw["ridge_lambda"],
        "stage1_sensitivity": stage1_results,
        "stage1_density_ranking": density_ranking,
        "stage1_coding_ranking": coding_ranking,
        "selected_density_values": top_densities,
        "selected_coding_values": top_codings,
        "stage2_interactions": stage2_results,
        "soho_candidates": stage2_results,
        "raw_candidates": raw_results,
        "outer_confirmation": outer_confirmation,
        "source_feature_metadata": metadata,
        "dataset_audit": audit,
        "soho_state_disclosure": (
            "SOHO replay is not exemplar-free: all historical frozen-backbone features and labels "
            "are learner state and are counted in every result"
        ),
        "warning": "ImageNet-R is a legacy processed split with 19 cross-split duplicate hashes"
        if dataset_key == "imagenetr" else None,
    }
    _atomic_json(output_dir / "selection.json", payload)
    print(
        f"SELECTION COMPLETE dataset={dataset_key} status={payload['status']} "
        f"soho={selected_soho['config']} raw_lambda={selected_raw['ridge_lambda']:g}",
        flush=True,
    )
    return payload


def lock_selection(protocol_path: Path, selection_root: Path, output_root: Path,
                   require_clean_git: bool) -> dict:
    protocol = _read_protocol(protocol_path)
    identities = _verify_method_identity(protocol)
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
        if payload["selected_soho_config"] not in _soho_candidates(protocol):
            raise ValueError(f"SOHO config outside locked search space for {key}")
        if float(payload["selected_raw_ridge_lambda"]) not in set(map(float, protocol["selection"]["raw_ridge_grid"])):
            raise ValueError(f"raw Ridge lambda outside locked grid for {key}")
        selections[key] = {"path": str(path), "sha256": _sha256_file(path)}
        selected[key] = {
            "soho_config": payload["selected_soho_config"],
            "raw_ridge_lambda": float(payload["selected_raw_ridge_lambda"]),
        }
    import subprocess
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    if require_clean_git and git_dirty:
        raise RuntimeError("final lock requires a clean Git worktree")
    record = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _sha256_file(protocol_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "method_identity": identities,
        "selection_files": selections,
        "selected_hyperparameters": selected,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "test_tuning_allowed": False,
        "accuracy_based_early_stop": False,
    }
    record["authorization_id"] = _sha256_json(record)
    path = output_root / "authorization.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        immutable = (
            "protocol_sha256", "runner_sha256", "method_identity", "selection_files",
            "selected_hyperparameters", "git_commit",
        )
        if any(previous.get(field) != record.get(field) for field in immutable):
            raise RuntimeError("existing authorization belongs to different code/selection")
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


def extract_test(*, protocol_path: Path, dataset_key: str, authorization_path: Path,
                 selection_root: Path, feature_cache_dir: Path, dataset_root: str,
                 checkpoint_path: str, device_name: str, batch_size: int, num_workers: int) -> dict:
    protocol = _read_protocol(protocol_path)
    authorization = _validate_authorization(authorization_path, protocol_path, selection_root)
    dataset, backbone = protocol["datasets"][dataset_key], protocol["backbone"]
    test_path, metadata_path = feature_cache_dir / "test.pt", feature_cache_dir / "metadata.json"
    if test_path.is_file():
        _, test, metadata = _validate_cache(feature_cache_dir, protocol, dataset_key, require_test=True)
        if metadata.get("authorization_id") != authorization["authorization_id"]:
            raise RuntimeError("existing test cache belongs to a different authorization")
        print(f"TEST CACHE RESTORED {dataset_key} shape={tuple(test['features'].shape)}", flush=True)
        return {"status": "restored", "test_sha256": _sha256_file(test_path)}
    _validate_cache(feature_cache_dir, protocol, dataset_key, require_test=False)
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
    for task, loader in enumerate(test_loaders):
        values, targets = feature_extract(model, loader, device)
        features.append(values.cpu()); labels.append(targets.cpu())
        print(f"TEST EXTRACT {dataset_key} task={task+1}/{dataset['num_tasks']} samples={len(targets)}", flush=True)
    packed = {"features": torch.cat(features), "labels": torch.cat(labels)}
    if (
        tuple(packed["features"].shape) != (dataset["test_samples"], backbone["feature_dim"])
        or tuple(packed["labels"].shape) != (dataset["test_samples"],)
        or not bool(torch.isfinite(packed["features"]).all())
    ):
        raise ValueError("test extraction tensor contract mismatch")
    _atomic_torch(test_path, packed)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "test_shape": list(packed["features"].shape),
        "test_labels_shape": list(packed["labels"].shape),
        "test_features_materialized": True,
        "authorization_id": authorization["authorization_id"],
        "test_sha256": _sha256_file(test_path),
    })
    _atomic_json(metadata_path, metadata)
    print(f"TEST CACHE COMPLETE {dataset_key} shape={tuple(packed['features'].shape)}", flush=True)
    return {"status": "complete", "test_sha256": metadata["test_sha256"]}


def evaluate_dataset(*, protocol_path: Path, dataset_key: str, selection_root: Path,
                     authorization_path: Path, feature_cache_dir: Path, output_root: Path,
                     dataset_audit_path: Path | None, device_name: str) -> dict:
    protocol = _read_protocol(protocol_path)
    _verify_method_identity(protocol)
    authorization = _validate_authorization(authorization_path, protocol_path, selection_root)
    dataset = protocol["datasets"][dataset_key]
    audit = _validate_dataset_audit(dataset_audit_path, dataset_key, dataset)
    train, test, metadata = _validate_cache(feature_cache_dir, protocol, dataset_key, require_test=True)
    selection_path = selection_root / dataset_key / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    soho_config = selection["selected_soho_config"]
    raw_ridge = float(selection["selected_raw_ridge_lambda"])
    stream = {
        "features": torch.cat((train["features"], test["features"])),
        "labels": torch.cat((train["labels"], test["labels"])),
    }
    test_offset = len(train["labels"])
    output_dir = output_root / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "protocol_sha256": _sha256_file(protocol_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "authorization_id": authorization["authorization_id"],
        "selection_sha256": _sha256_file(selection_path),
        "train_sha256": _sha256_file(feature_cache_dir / "train.pt"),
        "test_sha256": _sha256_file(feature_cache_dir / "test.pt"),
    }
    seed_results = []
    for replicate_index, replicate in enumerate(protocol["final_evaluation"]["replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(dataset["num_classes"]), dataset["num_classes"]
        )
        train_parts = _task_parts(train["labels"], class_order, dataset["num_tasks"])
        test_parts = _task_parts(test["labels"], class_order, dataset["num_tasks"], test_offset)
        methods = {}
        for method in METHODS:
            context = {
                **source, "dataset_key": dataset_key, "replicate": replicate,
                "class_order": class_order, "method": method,
                "soho_config": soho_config, "raw_ridge": raw_ridge,
            }
            methods[method] = _unit(
                output_dir / "units" / f"final_r{replicate_index}_{method}.json",
                context,
                lambda method=method, replicate=replicate, train_parts=train_parts, test_parts=test_parts:
                    _evaluate(
                        method, protocol, dataset, replicate["projection_seed"], stream,
                        train_parts, test_parts, soho_config, raw_ridge, device_name, True,
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
        {"replicate_index": item["replicate_index"], "method": method, "status": result.get("status")}
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
            "raw_ridge_lambda": raw_ridge,
            "fly_policy": "per-task GCV over locked exponents",
            "soho_ridge_policy": "per-task replay-sample GCV over locked exponents",
        },
        "source_feature_metadata": metadata,
        "dataset_audit": audit,
        "environment": _environment(device_name),
        "feature_cache_disk_bytes": sum(path.stat().st_size for path in feature_cache_dir.iterdir() if path.is_file()),
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
    _atomic_json(output_dir / "final_results.json", payload)
    print(f"DATASET COMPLETE {dataset_key} status={payload['status']}", flush=True)
    return payload


def _mean_std_ci(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "sample_std": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "sample_std": None, "ci95_low": None, "ci95_high": None}
    std = statistics.stdev(values)
    critical = T_CRITICAL_95_DF5 if len(values) == 6 else 1.96
    half = critical * std / math.sqrt(len(values))
    return {"n": len(values), "mean": mean, "sample_std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def summarize(protocol_path: Path, output_root: Path) -> dict:
    protocol = _read_protocol(protocol_path)
    rows, curves, paired = [], [], {}
    summaries = {}
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
                metric: _mean_std_ci([float(result[metric]) for result in results])
                for metric in (
                    "final_accuracy", "average_incremental_accuracy", "forgetting",
                    "persistent_state_bytes", "total_update_seconds", "total_inference_seconds",
                    "peak_runtime_memory_bytes",
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
                        "dataset": key, "method": method, "replicate_index": replicate_index,
                        "task": task, "task_fraction": task / len(result["stage_accuracy"]),
                        "average_seen_accuracy": value,
                    })
        for comparison in (("soho_replay_fidelity", "flycl_fidelity"), ("soho_replay_fidelity", "raw_ridge")):
            label = f"{comparison[0]}_minus_{comparison[1]}"
            differences = [
                item["methods"][comparison[0]]["average_incremental_accuracy"]
                - item["methods"][comparison[1]]["average_incremental_accuracy"]
                for item in payload["seed_results"]
                if item["methods"][comparison[0]].get("status") == "complete"
                and item["methods"][comparison[1]].get("status") == "complete"
            ]
            paired.setdefault(key, {})[label] = _mean_std_ci(differences)
    summary = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "status": "REPORTED_WITHOUT_ACCURACY_GATE",
        "uses_test_set": True,
        "dataset_method_summaries": summaries,
        "paired_aia_differences": paired,
        "soho_exemplar_free": False,
        "soho_state_disclosure": "historical frozen-backbone features and labels are persistent learner state",
        "test_reuse_disclosure": (
            "These repository test splits were used by earlier project phases; this is a locked comparative "
            "evaluation, not a first-use untouched held-out study."
        ),
        "imagenetr_disclosure": "legacy processed split with 19 duplicate hashes; not content-disjoint",
    }
    _atomic_json(output_root / "final_summary.json", summary)
    for path, data in ((output_root / "metrics_summary.csv", rows), (output_root / "task_curves.csv", curves)):
        fields = sorted({field for row in data for field in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--protocol", required=True); select.add_argument("--dataset-key", choices=DATASET_KEYS, required=True)
    select.add_argument("--feature-cache-dir", required=True); select.add_argument("--output-root", required=True)
    select.add_argument("--dataset-audit"); select.add_argument("--device", default="cpu")
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
    evaluate.add_argument("--feature-cache-dir", required=True); evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--dataset-audit"); evaluate.add_argument("--device", default="cpu")
    report = sub.add_parser("summarize")
    report.add_argument("--protocol", required=True); report.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    protocol_path = Path(args.protocol).resolve()
    if args.command == "select":
        select_dataset(
            protocol_path=protocol_path, dataset_key=args.dataset_key,
            feature_cache_dir=Path(args.feature_cache_dir).resolve(), output_root=Path(args.output_root).resolve(),
            dataset_audit_path=None if args.dataset_audit is None else Path(args.dataset_audit).resolve(),
            device_name=args.device,
        )
    elif args.command == "lock":
        lock_selection(protocol_path, Path(args.selection_root).resolve(), Path(args.output_root).resolve(), args.require_clean_git)
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
            feature_cache_dir=Path(args.feature_cache_dir).resolve(), output_root=Path(args.output_root).resolve(),
            dataset_audit_path=None if args.dataset_audit is None else Path(args.dataset_audit).resolve(),
            device_name=args.device,
        )
    else:
        summarize(protocol_path, Path(args.output_root).resolve())


if __name__ == "__main__":
    main()
