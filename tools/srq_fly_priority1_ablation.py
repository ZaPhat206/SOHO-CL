"""Train-only, isolated six-way SRQ-FLY Priority-1 ablation.

The runner never loads ``test.pt``.  Every method executes in a fresh process,
which makes CUDA peak-memory measurements comparable and resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.srq_fly_optimized import (
    DirectInt8GramLearner, SquareRootFLYLearner, projected_srq_state_bytes,
)
from tools import srq_fly_d0 as d0
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.srq_fly_d2_state_match import exact_fly_state_bytes
from tools.twa_fly_pilot import _prepare_code_cache, _sha256_file


METHODS = (
    "exact_fly_10000",
    "srq_int8_optimized",
    "sqrt_float16",
    "direct_int8_gram",
    "state_matched_exact_fly",
    "raw_ridge",
)
# Commit e44cb55 completed three expensive workers before the direct-Gram
# control exposed its expected SPD failure.  The evaluation semantics and
# learner/storage identities of those completed units are unchanged by the
# failure-recording patch, so their runner hash is explicitly auditable and
# resumable instead of forcing a costly rerun.
COMPATIBLE_COMPLETED_RUNNER_SHA256 = {
    "4f132747d879e42c71c7ce4401cd3b15685fe470cbeb8420019bd6e01d844cd1",
}
TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name",
    "checkpoint_sha256", "feature_dim", "seed", "num_classes", "num_tasks",
    "validation_fraction", "statistics_dtype", "solver_dtype",
    "fly_ridge_lambda", "raw_ridge_lambda", "srq_update_backend",
    "srq_update_panel_size",
    "large_representation", "state_matched_representation", "storage", "gates",
}
GATE_KEYS = {
    "maximum_solver_relative_residual", "maximum_srq_gap_to_exact_fly_pp",
    "maximum_float16_gap_to_exact_fly_pp",
    "maximum_state_match_error_fraction",
    "maximum_srq_state_fraction_of_exact", "minimum_system_update_speedup",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("Priority-1 config keys/schema mismatch")
    if config["seed"] != 2025:
        raise ValueError("new SRQ-FLY protocols require seed 2025")
    if (
        config["feature_dim"] <= 0 or config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
        or not 0 < config["validation_fraction"] < 1
    ):
        raise ValueError("invalid class/task/validation protocol")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("Priority-1 state matching is locked to float32")
    if config["fly_ridge_lambda"] <= 0 or config["raw_ridge_lambda"] <= 0:
        raise ValueError("Ridge parameters must be positive")
    if config["srq_update_backend"] != "blocked_qr":
        raise ValueError("Priority-1 optimized backend identity changed")
    if config["srq_update_panel_size"] <= 0:
        raise ValueError("Priority-1 update panel size must be positive")
    for key in ("large_representation", "state_matched_representation"):
        representation = config[key]
        if set(representation) != d0.REPRESENTATION_KEYS:
            raise ValueError(f"{key} fields mismatch")
        if min(
            representation[name] for name in (
                "expand_dim", "synaptic_degree", "encode_batch_size",
                "evaluation_batch_size",
            )
        ) <= 0 or not 0 < representation["coding_level"] <= 1:
            raise ValueError(f"invalid {key}")
    if set(config["storage"]) != d0.STORAGE_KEYS:
        raise ValueError("storage fields mismatch")
    if set(config["gates"]) != GATE_KEYS or any(
        float(value) < 0 for value in config["gates"].values()
    ):
        raise ValueError("gate fields mismatch")
    if config["gates"]["maximum_solver_relative_residual"] <= 0:
        raise ValueError("solver tolerance must be positive")
    return config


def _d0_config(config: dict) -> dict:
    """Adapter for the already-tested exact/raw evaluators."""
    return {
        "statistics_dtype": config["statistics_dtype"],
        "solver_dtype": config["solver_dtype"],
        "ridge_lambda": config["fly_ridge_lambda"],
        "raw_ridge_lambda": config["raw_ridge_lambda"],
        "seed": config["seed"],
        "large_representation": config["large_representation"],
        "storage": config["storage"],
    }


def _code_cache_config(config: dict, representation: dict) -> dict:
    return {
        "seed": config["seed"],
        "num_classes": config["num_classes"],
        "representation": dict(representation),
        "statistics_dtype": config["statistics_dtype"],
        "raw_ridge_lambda": config["raw_ridge_lambda"],
        "solver_tolerance": config["gates"]["maximum_solver_relative_residual"],
        "solver_max_iterations": 100,
    }


def _load_stream(
    *, config: dict, feature_cache_dir: Path, code_cache_dir: Path | None,
    representation: dict | None, device_name: str,
):
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("held-out test.pt must remain hidden")
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint SHA-256 mismatch")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(
        range(config["num_classes"])
    ):
        raise ValueError("training labels do not match locked classes")
    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"], task_indices, config["seed"], config["validation_fraction"]
    )
    code_cache = None
    if representation is not None:
        if code_cache_dir is None:
            raise ValueError("code cache path is required for a FLY method")
        code_cache = _prepare_code_cache(
            train=train,
            train_sha256=_sha256_file(feature_cache_dir / "train.pt"),
            cache_dir=code_cache_dir,
            config=_code_cache_config(config, representation),
            device=device_name,
        )
    return train, class_order, training_parts, validation_parts, code_cache


def _new_optimized(config: dict, method: str, feature_dim: int, projection, device):
    representation = config["large_representation"]
    kwargs = {
        "feature_dim": feature_dim,
        "expand_dim": int(representation["expand_dim"]),
        "synaptic_degree": int(representation["synaptic_degree"]),
        "coding_level": float(representation["coding_level"]),
        "ridge_lambda": float(config["fly_ridge_lambda"]),
        "block_size": int(config["storage"]["block_size"]),
        "group_size": int(config["storage"]["group_size"]),
        "seed": int(config["seed"]),
        "device": device,
        "statistics_dtype": torch.float32,
        "solver_dtype": torch.float32,
        "projection": projection,
    }
    if method == "direct_int8_gram":
        return DirectInt8GramLearner(**kwargs)
    storage_mode = "float16" if method == "sqrt_float16" else "int8"
    return SquareRootFLYLearner(
        storage_mode=storage_mode,
        update_backend=config["srq_update_backend"],
        update_panel_size=int(config["srq_update_panel_size"]),
        **kwargs,
    )


def _evaluate_optimized(
    *, method: str, config: dict, train: dict, code_cache,
    training_parts: list[torch.Tensor], validation_parts: list[torch.Tensor],
    device: torch.device,
) -> dict:
    code_indices, code_values, _, projection = code_cache
    learner = _new_optimized(
        config, method, int(train["features"].shape[1]), projection, device
    )
    stage_accuracy, diagnostics = [], []
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        task_started = time.perf_counter()
        codes = d0._dense_codes(
            code_indices[indices], code_values[indices], learner.expand_dim,
            device=device, dtype=learner.statistics_dtype,
        )
        update_started = time.perf_counter()
        try:
            learner.update_codes(codes, train["labels"][indices])
        except RuntimeError as error:
            expected_direct_failure = (
                method == "direct_int8_gram"
                and str(error)
                == "compressed Ridge system is not numerically positive definite"
            )
            if not expected_direct_failure:
                raise
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            failure_seconds = time.perf_counter() - update_started
            print(
                f"NUMERICAL_FAILURE method={method} task={task + 1} "
                f"reason={error}",
                flush=True,
            )
            return {
                "method": method,
                "status": "numerical_failure",
                "uses_test_set": False,
                "exemplar_free": True,
                "failure_type": "non_positive_definite_quantized_gram",
                "failure_message": str(error),
                "failed_task": task + 1,
                "completed_tasks": task,
                "validation_average_accuracy": None,
                "stage_accuracy": stage_accuracy,
                "persistent_state_bytes": learner.persistent_state_bytes(),
                "maximum_solver_relative_residual": None,
                "total_update_seconds": sum(
                    row["update_seconds"] for row in diagnostics
                ) + failure_seconds,
                "analytic_and_validation_seconds": time.perf_counter() - started,
                "task_diagnostics": diagnostics,
            }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - update_started
        del codes
        accuracy = d0._stage_code_accuracy(
            learner.weights, learner.class_ids, validation_parts, task,
            code_indices, code_values, train["labels"], learner.expand_dim,
            int(config["large_representation"]["evaluation_batch_size"]),
        )
        stage_accuracy.append(accuracy)
        diagnostics.append({
            "task": task + 1,
            "validation_accuracy": accuracy,
            "update_seconds": update_seconds,
            "stage_seconds": time.perf_counter() - task_started,
            "persistent_state_bytes": learner.persistent_state_bytes(),
            "solver_relative_residual": learner.diagnostics[
                "solver_relative_residual"
            ],
        })
        print(
            f"TASK method={method} {task + 1}/{len(training_parts)} "
            f"AA={accuracy:.4f} update={update_seconds:.3f}s",
            flush=True,
        )
    return {
        "method": method,
        "status": "complete",
        "uses_test_set": False,
        "exemplar_free": True,
        "validation_average_accuracy": sum(stage_accuracy) / len(stage_accuracy),
        "stage_accuracy": stage_accuracy,
        "persistent_state_bytes": learner.persistent_state_bytes(),
        "maximum_solver_relative_residual": max(
            row["solver_relative_residual"] for row in diagnostics
        ),
        "total_update_seconds": sum(row["update_seconds"] for row in diagnostics),
        "analytic_and_validation_seconds": time.perf_counter() - started,
        "task_diagnostics": diagnostics,
    }


def run_worker(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    method = args.method
    large = config["large_representation"]
    matched = config["state_matched_representation"]
    representation = None if method == "raw_ridge" else (
        matched if method == "state_matched_exact_fly" else large
    )
    cache_path = None if representation is None else Path(
        args.matched_code_cache_dir
        if method == "state_matched_exact_fly" else args.large_code_cache_dir
    ).resolve()
    train, class_order, training_parts, validation_parts, code_cache = _load_stream(
        config=config,
        feature_cache_dir=Path(args.feature_cache_dir).resolve(),
        code_cache_dir=cache_path,
        representation=representation,
        device_name=args.device,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    adapted = _d0_config(config)
    if method == "exact_fly_10000":
        result = d0._evaluate_exact(
            name=method, config=adapted, representation=large, train=train,
            code_indices=code_cache[0], code_values=code_cache[1],
            projection=code_cache[3], training_parts=training_parts,
            validation_parts=validation_parts, device=device,
        )
    elif method == "state_matched_exact_fly":
        result = d0._evaluate_exact(
            name=method, config=adapted, representation=matched, train=train,
            code_indices=code_cache[0], code_values=code_cache[1],
            projection=code_cache[3], training_parts=training_parts,
            validation_parts=validation_parts, device=device,
        )
    elif method == "raw_ridge":
        result = d0._evaluate_raw(
            config=adapted, train=train, training_parts=training_parts,
            validation_parts=validation_parts, device=device,
        )
    else:
        result = _evaluate_optimized(
            method=method, config=config, train=train, code_cache=code_cache,
            training_parts=training_parts, validation_parts=validation_parts,
            device=device,
        )
    result.update(
        class_order=class_order,
        config_sha256=_sha256(config_path),
        source_identity={
            "runner": _sha256(Path(__file__).resolve()),
            "optimized_learner": _sha256(
                ROOT / "methods/srq_fly_optimized/learner.py"
            ),
            "optimized_storage": _sha256(
                ROOT / "methods/srq_fly_optimized/storage.py"
            ),
        },
        peak_cuda_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        ),
        peak_cuda_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda" else None
        ),
        peak_memory_scope="method construction, analytic update, and validation; frozen feature extraction excluded",
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _validate_system_result(config: dict, path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass" or payload.get("uses_test_set") is not False:
        raise ValueError("system optimization benchmark did not pass")
    gates = payload.get("gates", {})
    if not gates.get("blocked_qr_backend_within_tolerance"):
        raise ValueError("blocked QR backend did not pass predictor gate")
    selected = payload.get("selected_update_backend", {})
    if selected != {
        "name": config["srq_update_backend"],
        "panel_size": config["srq_update_panel_size"],
    }:
        raise ValueError("system benchmark backend identity does not match protocol")
    speedup = payload.get("speedup_over_locked", {}).get(
        "optimized_blocked_qr_srq_int8", 0.0
    )
    if speedup < config["gates"]["minimum_system_update_speedup"]:
        raise ValueError("optimized blocked backend did not meet the update-speed gate")
    return payload


def run_driver(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("held-out test.pt must remain hidden")
    system = _validate_system_result(config, Path(args.system_benchmark_result).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for method in METHODS:
        output = output_dir / f"{method}.json"
        expected_source = {
            "runner": _sha256(Path(__file__).resolve()),
            "optimized_learner": _sha256(
                ROOT / "methods/srq_fly_optimized/learner.py"
            ),
            "optimized_storage": _sha256(
                ROOT / "methods/srq_fly_optimized/storage.py"
            ),
        }
        if output.is_file():
            restored = json.loads(output.read_text(encoding="utf-8"))
            restored_source = restored.get("source_identity", {})
            source_compatible = (
                restored_source.get("runner")
                in {expected_source["runner"], *COMPATIBLE_COMPLETED_RUNNER_SHA256}
                and restored_source.get("optimized_learner")
                == expected_source["optimized_learner"]
                and restored_source.get("optimized_storage")
                == expected_source["optimized_storage"]
            )
            resumable_status = (
                restored.get("status") == "complete"
                or (
                    method == "direct_int8_gram"
                    and restored.get("status") == "numerical_failure"
                )
            )
            if (
                resumable_status
                and restored.get("method") == method
                and restored.get("uses_test_set") is False
                and restored.get("config_sha256") == _sha256(config_path)
                and source_compatible
            ):
                results.append(restored)
                print(f"RESUME isolated ablation method={method}", flush=True)
                continue
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()), "worker",
            "--config", str(config_path), "--feature-cache-dir", str(feature_cache_dir),
            "--large-code-cache-dir", str(Path(args.large_code_cache_dir).resolve()),
            "--matched-code-cache-dir", str(Path(args.matched_code_cache_dir).resolve()),
            "--output", str(output), "--method", method, "--device", args.device,
        ]
        print(f"START isolated ablation method={method}", flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            raise RuntimeError(f"ablation worker failed: {method}")
        results.append(json.loads(output.read_text(encoding="utf-8")))
        print(f"DONE isolated ablation method={method}", flush=True)
    by_method = {row["method"]: row for row in results}
    exact = by_method["exact_fly_10000"]
    srq = by_method["srq_int8_optimized"]
    float16 = by_method["sqrt_float16"]
    matched = by_method["state_matched_exact_fly"]
    direct_gram = by_method["direct_int8_gram"]
    projected = projected_srq_state_bytes(
        feature_dim=int(config["feature_dim"]),
        expand_dim=int(config["large_representation"]["expand_dim"]),
        synaptic_degree=int(config["large_representation"]["synaptic_degree"]),
        num_classes=int(config["num_classes"]),
        block_size=int(config["storage"]["block_size"]),
        group_size=int(config["storage"]["group_size"]),
    )
    expected_matched = exact_fly_state_bytes(
        feature_dim=int(config["feature_dim"]),
        expand_dim=int(config["state_matched_representation"]["expand_dim"]),
        synaptic_degree=int(config["state_matched_representation"]["synaptic_degree"]),
        num_classes=int(config["num_classes"]),
    )
    state_error = abs(matched["persistent_state_bytes"] - srq["persistent_state_bytes"]) / srq[
        "persistent_state_bytes"
    ]
    gates_config = config["gates"]
    primary_methods = set(METHODS) - {"direct_int8_gram"}
    primary_results = [row for row in results if row["method"] in primary_methods]
    gates = {
        "all_primary_methods_complete": len(primary_results) == len(primary_methods)
        and all(row["status"] == "complete" for row in primary_results),
        "all_methods_accounted_for": len(results) == len(METHODS)
        and direct_gram["status"] in {"complete", "numerical_failure"},
        "heldout_test_remained_hidden": not (feature_cache_dir / "test.pt").exists(),
        "system_update_gate_passed": system["status"] == "pass",
        "primary_methods_numerically_stable": max(
            row["maximum_solver_relative_residual"] for row in primary_results
        ) <= gates_config["maximum_solver_relative_residual"],
        "srq_within_accuracy_gate": exact["validation_average_accuracy"]
        - srq["validation_average_accuracy"]
        <= gates_config["maximum_srq_gap_to_exact_fly_pp"],
        "float16_within_accuracy_gate": exact["validation_average_accuracy"]
        - float16["validation_average_accuracy"]
        <= gates_config["maximum_float16_gap_to_exact_fly_pp"],
        "state_match_runtime_gate": state_error
        <= gates_config["maximum_state_match_error_fraction"],
        "state_match_formula_gate": matched["persistent_state_bytes"] == expected_matched,
        "srq_state_fraction_gate": srq["persistent_state_bytes"]
        / exact["persistent_state_bytes"]
        <= gates_config["maximum_srq_state_fraction_of_exact"],
    }
    summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_REVIEW_PRIORITY1" if all(gates.values()) else "STOP_PRIORITY1",
        "uses_test_set": False,
        "held_out_test_authorized": False,
        "config_sha256": _sha256(config_path),
        "system_benchmark_sha256": _sha256(Path(args.system_benchmark_result).resolve()),
        "source_identity": {
            "runner": _sha256(Path(__file__).resolve()),
            "optimized_learner": _sha256(ROOT / "methods/srq_fly_optimized/learner.py"),
            "optimized_storage": _sha256(ROOT / "methods/srq_fly_optimized/storage.py"),
            "exact_control_runner": _sha256(ROOT / "tools/srq_fly_d0.py"),
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        "projected_state": projected,
        "runtime_state_match_error_fraction": state_error,
        "control_observations": {
            "direct_int8_gram_status": direct_gram["status"],
            "direct_int8_gram_spd_preserved": direct_gram["status"] == "complete",
            "direct_int8_gram_failure_type": direct_gram.get("failure_type"),
            "direct_int8_gram_failed_task": direct_gram.get("failed_task"),
        },
        "results": results,
        "gates": gates,
    }
    (output_dir / "priority1_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    driver = subparsers.add_parser("run")
    for item in (worker, driver):
        item.add_argument("--config", required=True)
        item.add_argument("--feature-cache-dir", required=True)
        item.add_argument("--large-code-cache-dir", required=True)
        item.add_argument("--matched-code-cache-dir", required=True)
        item.add_argument("--device", default="cpu")
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--output", required=True)
    driver.add_argument("--system-benchmark-result", required=True)
    driver.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
