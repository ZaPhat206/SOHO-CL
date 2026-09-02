"""Real CIFAR-100 train-only equivalence for the final SRQ runtime backend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.srq_fly_optimized import SquareRootFLYLearner
from tools import srq_fly_d0 as d0
from tools import srq_fly_priority1_ablation as p1


METHODS = ("priority2b_batch64", "implicit_ridge_batch64")
TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name",
    "checkpoint_sha256", "feature_dim", "seed", "num_classes", "num_tasks",
    "validation_fraction", "statistics_dtype", "solver_dtype", "ridge_lambda",
    "representation", "storage", "update_panel_size",
    "quantization_batch_blocks", "probe_rows", "gates",
}
GATE_KEYS = {
    "maximum_stage_accuracy_gap_pp", "maximum_relative_logit_drift",
    "maximum_solver_relative_residual",
    "maximum_peak_allocated_ratio_to_priority2b",
    "maximum_update_ratio_to_priority2b",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, str]:
    return {
        "runner": _sha256(Path(__file__).resolve()),
        "optimized_learner": _sha256(ROOT / "methods/srq_fly_optimized/learner.py"),
        "optimized_storage": _sha256(ROOT / "methods/srq_fly_optimized/storage.py"),
        "stream_loader": _sha256(ROOT / "tools/srq_fly_priority1_ablation.py"),
    }


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("Priority-2D config keys/schema mismatch")
    if set(config["gates"]) != GATE_KEYS:
        raise ValueError("Priority-2D gate fields mismatch")
    if config["dataset"] != "CIFAR-100" or config["feature_dim"] != 768:
        raise ValueError("Priority-2D is locked to CIFAR-100 ViT features")
    if config["seed"] != 2025 or config["num_tasks"] != 10:
        raise ValueError("Priority-2D split identity changed")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("Priority-2D numerical dtype changed")
    if set(config["representation"]) != d0.REPRESENTATION_KEYS:
        raise ValueError("Priority-2D representation fields mismatch")
    if set(config["storage"]) != d0.STORAGE_KEYS:
        raise ValueError("Priority-2D storage fields mismatch")
    if int(config["quantization_batch_blocks"]) != 64:
        raise ValueError("Priority-2D must use selected batch 64")
    if min(float(value) for value in config["gates"].values()) <= 0:
        raise ValueError("Priority-2D gates must be positive")
    return config


def _learner(config: dict, method: str, projection, device: torch.device):
    representation = config["representation"]
    return SquareRootFLYLearner(
        feature_dim=int(config["feature_dim"]),
        expand_dim=int(representation["expand_dim"]),
        synaptic_degree=int(representation["synaptic_degree"]),
        coding_level=float(representation["coding_level"]),
        ridge_lambda=float(config["ridge_lambda"]),
        block_size=int(config["storage"]["block_size"]),
        group_size=int(config["storage"]["group_size"]),
        seed=int(config["seed"]),
        device=device,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        projection=projection,
        storage_mode="int8",
        update_backend="blocked_qr",
        update_panel_size=int(config["update_panel_size"]),
        first_update_backend=(
            "implicit_ridge_qr"
            if method == "implicit_ridge_batch64" else "gram_cholesky"
        ),
        quantization_backend="streaming",
        quantization_batch_blocks=int(config["quantization_batch_blocks"]),
    )


def _probe_indices(validation_parts: list[torch.Tensor], rows: int) -> torch.Tensor:
    combined = torch.cat(validation_parts)
    if len(combined) < rows:
        raise ValueError("not enough validation rows for locked probe")
    return combined[:rows]


def run_worker(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    feature_cache = Path(args.feature_cache_dir).resolve()
    if (feature_cache / "test.pt").exists():
        raise RuntimeError("held-out test.pt must remain hidden")
    train, class_order, training_parts, validation_parts, code_cache = p1._load_stream(
        config={
            **config,
            "large_representation": config["representation"],
            "raw_ridge_lambda": 1.0,
            "gates": {
                "maximum_solver_relative_residual": config["gates"][
                    "maximum_solver_relative_residual"
                ]
            },
        },
        feature_cache_dir=feature_cache,
        code_cache_dir=Path(args.code_cache_dir).resolve(),
        representation=config["representation"],
        device_name=args.device,
    )
    code_indices, code_values, _, projection = code_cache
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    learner = _learner(config, args.method, projection, device)
    stage_accuracy = []
    task_diagnostics = []
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        codes = d0._dense_codes(
            code_indices[indices], code_values[indices], learner.expand_dim,
            device=device, dtype=learner.statistics_dtype,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_started = time.perf_counter()
        learner.update_codes_consuming(codes, train["labels"][indices])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - update_started
        del codes
        accuracy = d0._stage_code_accuracy(
            learner.weights, learner.class_ids, validation_parts, task,
            code_indices, code_values, train["labels"], learner.expand_dim,
            int(config["representation"]["evaluation_batch_size"]),
        )
        stage_accuracy.append(accuracy)
        task_diagnostics.append({
            "task": task + 1,
            "validation_accuracy": accuracy,
            "update_seconds": update_seconds,
            "persistent_state_bytes": learner.persistent_state_bytes(),
            "solver_relative_residual": learner.diagnostics[
                "solver_relative_residual"
            ],
        })
        print(
            f"TASK method={args.method} {task+1}/{len(training_parts)} "
            f"AA={accuracy:.4f} update={update_seconds:.3f}s", flush=True,
        )
    probe_indices = _probe_indices(validation_parts, int(config["probe_rows"]))
    probe_codes = d0._dense_codes(
        code_indices[probe_indices], code_values[probe_indices], learner.expand_dim,
        device=device, dtype=learner.statistics_dtype,
    )
    probe = learner.predict_logits_from_codes(probe_codes).detach().cpu()
    probe_path = Path(args.probe_output).resolve()
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe, probe_path)
    result = {
        "schema_version": 1,
        "status": "complete",
        "method": args.method,
        "uses_test_set": False,
        "class_order": class_order,
        "validation_average_accuracy": sum(stage_accuracy) / len(stage_accuracy),
        "stage_accuracy": stage_accuracy,
        "persistent_state_bytes": learner.persistent_state_bytes(),
        "maximum_solver_relative_residual": max(
            row["solver_relative_residual"] for row in task_diagnostics
        ),
        "total_update_seconds": sum(row["update_seconds"] for row in task_diagnostics),
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda" else None
        ),
        "task_diagnostics": task_diagnostics,
        "config_sha256": _sha256(config_path),
        "source_identity": _source_identity(),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_driver(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    feature_cache = Path(args.feature_cache_dir).resolve()
    if (feature_cache / "test.pt").exists():
        raise RuntimeError("held-out test.pt must remain hidden")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if args.require_clean_git and dirty:
        raise RuntimeError(f"repository must be clean before Priority-2D:\n{dirty}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results, probes = [], {}
    for method in METHODS:
        output = output_dir / f"{method}.json"
        probe = output_dir / f"{method}.probe.pt"
        resumable = False
        if output.is_file() and probe.is_file():
            cached = json.loads(output.read_text(encoding="utf-8"))
            resumable = (
                cached.get("status") == "complete"
                and cached.get("method") == method
                and cached.get("uses_test_set") is False
                and cached.get("config_sha256") == _sha256(config_path)
                and cached.get("source_identity") == _source_identity()
            )
        if not resumable:
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                "--config", str(config_path),
                "--feature-cache-dir", str(feature_cache),
                "--code-cache-dir", str(Path(args.code_cache_dir).resolve()),
                "--output", str(output), "--probe-output", str(probe),
                "--method", method, "--device", args.device,
            ]
            print(f"START isolated real-data method={method}", flush=True)
            if subprocess.run(command, cwd=ROOT).returncode:
                raise RuntimeError(f"Priority-2D worker failed: {method}")
            print(f"DONE isolated real-data method={method}", flush=True)
        else:
            print(f"RESUME isolated real-data method={method}", flush=True)
        results.append(json.loads(output.read_text(encoding="utf-8")))
        probes[method] = torch.load(probe, weights_only=True, map_location="cpu")
    by_method = {row["method"]: row for row in results}
    baseline = by_method["priority2b_batch64"]
    candidate = by_method["implicit_ridge_batch64"]
    denominator = max(float(torch.linalg.vector_norm(probes["priority2b_batch64"])), 1.0)
    drift = float(torch.linalg.vector_norm(
        probes["implicit_ridge_batch64"] - probes["priority2b_batch64"]
    )) / denominator
    stage_gap = max(abs(a-b) for a, b in zip(
        candidate["stage_accuracy"], baseline["stage_accuracy"]
    ))
    update_ratio = candidate["total_update_seconds"] / baseline["total_update_seconds"]
    peak_ratio = candidate["peak_cuda_allocated_bytes"] / baseline[
        "peak_cuda_allocated_bytes"
    ]
    gates_config = config["gates"]
    gates = {
        "both_methods_complete": all(row["status"] == "complete" for row in results),
        "heldout_test_remained_hidden": not (feature_cache / "test.pt").exists(),
        "stage_accuracy_within_tolerance": stage_gap
        <= gates_config["maximum_stage_accuracy_gap_pp"],
        "predictor_within_tolerance": drift
        <= gates_config["maximum_relative_logit_drift"],
        "persistent_state_unchanged": candidate["persistent_state_bytes"]
        == baseline["persistent_state_bytes"],
        "solver_stable": candidate["maximum_solver_relative_residual"]
        <= gates_config["maximum_solver_relative_residual"],
        "peak_memory_not_regressed": peak_ratio
        <= gates_config["maximum_peak_allocated_ratio_to_priority2b"],
        "update_time_not_regressed": update_ratio
        <= gates_config["maximum_update_ratio_to_priority2b"],
    }
    summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_FINAL_BACKEND" if all(gates.values()) else "STOP_PRIORITY2D",
        "uses_test_set": False,
        "held_out_test_authorized": False,
        "config_sha256": _sha256(config_path),
        "source_identity": _source_identity(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(dirty),
        "maximum_stage_accuracy_gap_pp": stage_gap,
        "relative_probe_logit_drift": drift,
        "update_ratio_to_priority2b": update_ratio,
        "peak_allocated_ratio_to_priority2b": peak_ratio,
        "results": results,
        "gates": gates,
    }
    (output_dir / "priority2d_equivalence_results.json").write_text(
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
        item.add_argument("--code-cache-dir", required=True)
        item.add_argument("--device", default="cpu")
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--output", required=True)
    worker.add_argument("--probe-output", required=True)
    driver.add_argument("--output-dir", required=True)
    driver.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args()
    if args.command == "worker":
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
