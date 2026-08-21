"""Locked five-task, train-only SRQ-FLY diagnostic.

This runner deliberately has no held-out evaluation mode. WTA code caches are
experiment infrastructure and are never included in learner checkpoints.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.srq_fly import DirectInt8GramLearner, SquareRootFLYLearner
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.tail_fly_phasea import (
    _dense_codes, _expand_cross, _git_provenance, _load_unit, _raw_accuracy,
    _save_unit, _stage_code_accuracy, _state_bytes, _targets, _unit_path,
)
from tools.twa_fly_pilot import (
    _prepare_code_cache, _sequence_sha256, _sha256_bytes, _sha256_file,
    _tensor_content_sha256,
)


TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name", "checkpoint_sha256",
    "seed", "num_classes", "num_tasks", "diagnostic_tasks", "validation_fraction",
    "statistics_dtype", "solver_dtype", "ridge_lambda", "raw_ridge_lambda",
    "large_representation", "compact_representation", "storage", "gates",
}
REPRESENTATION_KEYS = {
    "expand_dim", "synaptic_degree", "coding_level", "encode_batch_size",
    "evaluation_batch_size",
}
STORAGE_KEYS = {"block_size", "group_size"}
GATE_KEYS = {
    "maximum_solver_relative_residual", "maximum_gap_to_exact_fly_pp",
    "maximum_state_fraction_of_exact_fly",
}


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config["schema_version"] != 1:
        raise ValueError("config keys/schema mismatch")
    if config["seed"] != 2025:
        raise ValueError("new SRQ-FLY protocols require repository seed 2025")
    if (
        config["num_classes"] <= 1 or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
        or not 0 < config["diagnostic_tasks"] <= config["num_tasks"]
    ):
        raise ValueError("invalid class/task count")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    if config["statistics_dtype"] not in {"float32", "float64"} or config["solver_dtype"] not in {"float32", "float64"}:
        raise ValueError("invalid dtype")
    if config["ridge_lambda"] <= 0 or config["raw_ridge_lambda"] <= 0:
        raise ValueError("Ridge parameters must be positive")
    for name in ("large_representation", "compact_representation"):
        representation = config[name]
        if set(representation) != REPRESENTATION_KEYS:
            raise ValueError(f"{name} keys mismatch")
        if min(representation[key] for key in (
            "expand_dim", "synaptic_degree", "encode_batch_size", "evaluation_batch_size"
        )) <= 0 or not 0 < representation["coding_level"] <= 1:
            raise ValueError(f"invalid {name}")
    if config["large_representation"]["expand_dim"] <= config["compact_representation"]["expand_dim"]:
        raise ValueError("large representation must exceed compact representation")
    if set(config["storage"]) != STORAGE_KEYS or min(config["storage"].values()) <= 0:
        raise ValueError("invalid storage configuration")
    if set(config["gates"]) != GATE_KEYS:
        raise ValueError("gate keys mismatch")
    if (
        config["gates"]["maximum_solver_relative_residual"] <= 0
        or config["gates"]["maximum_gap_to_exact_fly_pp"] < 0
        or not 0 < config["gates"]["maximum_state_fraction_of_exact_fly"] <= 1
    ):
        raise ValueError("invalid gates")
    return config


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _cache_config(config: dict, representation_name: str) -> dict:
    return {
        "seed": config["seed"], "num_classes": config["num_classes"],
        "representation": dict(config[representation_name]),
        "statistics_dtype": "float32", "raw_ridge_lambda": 1.0,
        "solver_tolerance": 1e-5, "solver_max_iterations": 100,
    }


def _solve(system: torch.Tensor, cross: torch.Tensor) -> tuple[torch.Tensor, float]:
    factor, info = torch.linalg.cholesky_ex((system + system.T) * 0.5)
    if int(info.max().item()) != 0:
        raise RuntimeError("fixed-Ridge control Cholesky failed")
    weights = torch.cholesky_solve(cross, factor)
    residual = torch.linalg.vector_norm(system @ weights - cross) / max(
        float(torch.linalg.vector_norm(cross)), 1.0
    )
    return weights, float(residual)


def _evaluate_exact(
    *, name, config, representation, train, code_indices, code_values,
    projection, training_parts, validation_parts, device,
) -> dict:
    dtype = _dtype(config["statistics_dtype"])
    dimension = int(representation["expand_dim"])
    gram = torch.zeros((dimension, dimension), device=device, dtype=dtype)
    cross = torch.zeros((dimension, 0), device=device, dtype=dtype)
    counts = torch.zeros(0, device=device, dtype=dtype)
    class_ids, stage_accuracy, residuals = [], [], []
    weights = None
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        codes = _dense_codes(code_indices[indices], code_values[indices], dimension, device=device, dtype=dtype)
        labels = train["labels"][indices]
        new_ids = sorted(set(class_ids) | set(map(int, labels.tolist())))
        cross, counts = _expand_cross(cross, counts, class_ids, new_ids)
        class_ids = new_ids
        targets = _targets(labels, class_ids, device=device, dtype=dtype)
        gram += codes.T @ codes
        cross += codes.T @ targets
        counts += targets.sum(0)
        system = gram + float(config["ridge_lambda"]) * torch.eye(dimension, device=device, dtype=dtype)
        weights, residual = _solve(system, cross)
        residuals.append(residual)
        accuracy = _stage_code_accuracy(
            weights, class_ids, validation_parts, task, code_indices, code_values,
            train["labels"], dimension, int(representation["evaluation_batch_size"]),
        )
        stage_accuracy.append(accuracy)
        print(f"TASK {name} {task+1}/{len(training_parts)} AA={accuracy:.4f} residual={residual:.3e}", flush=True)
        del codes, system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "method": name, "status": "complete", "ridge_lambda": config["ridge_lambda"],
        "validation_average_accuracy": sum(stage_accuracy) / len(stage_accuracy),
        "stage_accuracy": stage_accuracy,
        "persistent_state_bytes": sum(_state_bytes(value) for value in (projection, gram, cross, counts, weights)),
        "maximum_solver_relative_residual": max(residuals),
        "seconds": time.perf_counter() - started, "uses_test_set": False,
        "exemplar_free": True,
    }


def _evaluate_raw(*, config, train, training_parts, validation_parts, device) -> dict:
    dtype = _dtype(config["statistics_dtype"])
    dimension = int(train["features"].shape[1])
    gram = torch.zeros((dimension, dimension), device=device, dtype=dtype)
    cross = torch.zeros((dimension, 0), device=device, dtype=dtype)
    counts = torch.zeros(0, device=device, dtype=dtype)
    class_ids, stage_accuracy, residuals = [], [], []
    weights = None
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        values = train["features"][indices].to(device=device, dtype=dtype)
        labels = train["labels"][indices]
        new_ids = sorted(set(class_ids) | set(map(int, labels.tolist())))
        cross, counts = _expand_cross(cross, counts, class_ids, new_ids)
        class_ids = new_ids
        targets = _targets(labels, class_ids, device=device, dtype=dtype)
        gram += values.T @ values
        cross += values.T @ targets
        counts += targets.sum(0)
        system = gram + float(config["raw_ridge_lambda"]) * torch.eye(dimension, device=device, dtype=dtype)
        weights, residual = _solve(system, cross)
        residuals.append(residual)
        accuracy = sum(
            _raw_accuracy(weights, class_ids, validation_parts[previous], train["features"], train["labels"], 256)
            for previous in range(task + 1)
        ) / (task + 1)
        stage_accuracy.append(accuracy)
        print(f"TASK raw_ridge {task+1}/{len(training_parts)} AA={accuracy:.4f}", flush=True)
    return {
        "method": "raw_ridge", "status": "complete", "ridge_lambda": config["raw_ridge_lambda"],
        "validation_average_accuracy": sum(stage_accuracy) / len(stage_accuracy),
        "stage_accuracy": stage_accuracy,
        "persistent_state_bytes": sum(_state_bytes(value) for value in (gram, cross, counts, weights)),
        "maximum_solver_relative_residual": max(residuals),
        "seconds": time.perf_counter() - started, "uses_test_set": False,
        "exemplar_free": True,
    }


def _new_learner(config, method, feature_dim, projection, device):
    representation, storage = config["large_representation"], config["storage"]
    kwargs = dict(
        feature_dim=feature_dim, expand_dim=int(representation["expand_dim"]),
        synaptic_degree=int(representation["synaptic_degree"]),
        coding_level=float(representation["coding_level"]),
        ridge_lambda=float(config["ridge_lambda"]), block_size=int(storage["block_size"]),
        group_size=int(storage["group_size"]), seed=int(config["seed"]), device=device,
        statistics_dtype=_dtype(config["statistics_dtype"]), solver_dtype=_dtype(config["solver_dtype"]),
        projection=projection,
    )
    if method == "direct_int8_gram":
        return DirectInt8GramLearner(**kwargs)
    return SquareRootFLYLearner(storage_mode={"sqrt_float16": "float16", "srq_int8": "int8"}[method], **kwargs)


def _evaluate_learner(
    *, method, config, train, code_indices, code_values, projection,
    training_parts, validation_parts, device,
) -> dict:
    learner = _new_learner(config, method, int(train["features"].shape[1]), projection, device)
    representation = config["large_representation"]
    stage_accuracy, task_diagnostics = [], []
    started = time.perf_counter()
    try:
        for task, indices in enumerate(training_parts):
            task_started = time.perf_counter()
            codes = _dense_codes(code_indices[indices], code_values[indices], learner.expand_dim, device=device, dtype=learner.statistics_dtype)
            learner.update_codes(codes, train["labels"][indices])
            del codes
            accuracy = _stage_code_accuracy(
                learner.weights, learner.class_ids, validation_parts, task,
                code_indices, code_values, train["labels"], learner.expand_dim,
                int(representation["evaluation_batch_size"]),
            )
            stage_accuracy.append(accuracy)
            task_diagnostics.append({
                "task": task + 1, "validation_accuracy": accuracy,
                "persistent_state_bytes": learner.persistent_state_bytes(),
                "solver_relative_residual": learner.diagnostics["solver_relative_residual"],
                "seconds": time.perf_counter() - task_started,
            })
            print(
                f"TASK {method} {task+1}/{len(training_parts)} AA={accuracy:.4f} "
                f"residual={learner.diagnostics['solver_relative_residual']:.3e} "
                f"state={learner.persistent_state_bytes()}B",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except (RuntimeError, torch.linalg.LinAlgError) as error:
        return {
            "method": method, "status": "solver_failed", "failure": f"{type(error).__name__}: {error}",
            "stage_accuracy": stage_accuracy, "task_diagnostics": task_diagnostics,
            "seconds": time.perf_counter() - started, "uses_test_set": False, "exemplar_free": True,
        }
    return {
        "method": method, "status": "complete",
        "validation_average_accuracy": sum(stage_accuracy) / len(stage_accuracy),
        "stage_accuracy": stage_accuracy, "persistent_state_bytes": learner.persistent_state_bytes(),
        "maximum_solver_relative_residual": max(item["solver_relative_residual"] for item in task_diagnostics),
        "task_diagnostics": task_diagnostics, "seconds": time.perf_counter() - started,
        "uses_test_set": False, "exemplar_free": True,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    if args.require_test_hidden and (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("held-out test.pt is visible")
    train, _, metadata = validate_cache(
        feature_cache_dir, argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]), load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint SHA-256 mismatch")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(range(config["num_classes"])):
        raise ValueError("training labels do not match locked classes")
    train_path = feature_cache_dir / "train.pt"
    train_sha256 = _sha256_file(train_path)
    caches = {}
    for name, argument, representation in (
        ("large", args.large_code_cache_dir, "large_representation"),
        ("compact", args.compact_code_cache_dir, "compact_representation"),
    ):
        caches[name] = _prepare_code_cache(
            train=train, train_sha256=train_sha256, cache_dir=Path(argument).resolve(),
            config=_cache_config(config, representation), device=args.device,
        )
    class_order = random.Random(config["seed"]).sample(list(range(config["num_classes"])), config["num_classes"])
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"], task_indices, config["seed"], config["validation_fraction"]
    )
    limit = int(config["diagnostic_tasks"])
    training_parts, validation_parts = training_parts[:limit], validation_parts[:limit]
    git = _git_provenance()
    context = {
        "config_sha256": _sha256_file(config_path), "train_sha256": train_sha256,
        "large_code_identity": caches["large"][2]["identity_sha256"],
        "compact_code_identity": caches["compact"][2]["identity_sha256"],
        "large_projection_sha256": _tensor_content_sha256(caches["large"][3]),
        "compact_projection_sha256": _tensor_content_sha256(caches["compact"][3]),
        "training_indices_sha256": _sequence_sha256(training_parts),
        "validation_indices_sha256": _sequence_sha256(validation_parts),
        "runner_git_commit": git["git_commit"], "runner_git_dirty": git["git_dirty"],
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "learner_source_sha256": _sha256_file(ROOT / "methods/srq_fly/learner.py"),
        "storage_source_sha256": _sha256_file(ROOT / "methods/srq_fly/storage.py"),
    }
    context_sha256 = _sha256_bytes(json.dumps(context, sort_keys=True).encode())
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results = []

    def unit(name, evaluator):
        path = _unit_path(output_dir, name)
        result = _load_unit(path, context_sha256)
        if result is None:
            print(f"START {name}", flush=True)
            result = _save_unit(path, context_sha256, evaluator())
            print(f"DONE {name} status={result['status']}", flush=True)
        else:
            print(f"RESUME {name}", flush=True)
        results.append(result)

    large = caches["large"]
    compact = caches["compact"]
    unit("exact_fly_10000", lambda: _evaluate_exact(
        name="exact_fly_10000", config=config, representation=config["large_representation"], train=train,
        code_indices=large[0], code_values=large[1], projection=large[3],
        training_parts=training_parts, validation_parts=validation_parts, device=device,
    ))
    unit("exact_fly_4096", lambda: _evaluate_exact(
        name="exact_fly_4096", config=config, representation=config["compact_representation"], train=train,
        code_indices=compact[0], code_values=compact[1], projection=compact[3],
        training_parts=training_parts, validation_parts=validation_parts, device=device,
    ))
    unit("raw_ridge", lambda: _evaluate_raw(
        config=config, train=train, training_parts=training_parts,
        validation_parts=validation_parts, device=device,
    ))
    for method in ("direct_int8_gram", "sqrt_float16", "srq_int8"):
        unit(method, lambda method=method: _evaluate_learner(
            method=method, config=config, train=train, code_indices=large[0], code_values=large[1],
            projection=large[3], training_parts=training_parts,
            validation_parts=validation_parts, device=device,
        ))

    by_name = {result["method"]: result for result in results}
    srq = by_name["srq_int8"]
    exact_large = by_name["exact_fly_10000"]
    exact_compact = by_name["exact_fly_4096"]
    complete = all(result["status"] == "complete" for result in results)
    gates = {
        "all_units_complete": complete,
        "heldout_test_remained_hidden": not (feature_cache_dir / "test.pt").exists(),
        "numerical_stability": complete and max(
            result["maximum_solver_relative_residual"] for result in results
        ) <= config["gates"]["maximum_solver_relative_residual"],
        "within_0.50pp_of_exact_fly_10000": complete and exact_large["validation_average_accuracy"] - srq["validation_average_accuracy"] <= config["gates"]["maximum_gap_to_exact_fly_pp"],
        "state_below_quarter_exact_fly_10000": complete and srq["persistent_state_bytes"] / exact_large["persistent_state_bytes"] <= config["gates"]["maximum_state_fraction_of_exact_fly"],
        "not_pareto_dominated_by_exact_fly_4096": complete and not (
            exact_compact["validation_average_accuracy"] >= srq["validation_average_accuracy"]
            and exact_compact["persistent_state_bytes"] <= srq["persistent_state_bytes"]
            and (exact_compact["validation_average_accuracy"] > srq["validation_average_accuracy"]
                 or exact_compact["persistent_state_bytes"] < srq["persistent_state_bytes"])
        ),
    }
    decision = "PASS_REVIEW_D0" if all(gates.values()) else "STOP_SRQ_FLY_D0"
    payload = {
        "schema_version": 1, "study_id": config["study_id"], "status": decision,
        "uses_test_set": False, "held_out_test_authorized": False,
        "diagnostic_tasks": limit, "provenance": context,
        "class_order": class_order, "results": results, "gates": gates,
    }
    path = output_dir / "d0_results.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(f"SRQ-FLY D0 {decision}", flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--large-code-cache-dir", required=True)
    parser.add_argument("--compact-code-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-test-hidden", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
