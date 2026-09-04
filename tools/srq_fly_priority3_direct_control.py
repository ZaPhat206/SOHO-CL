"""Train-only direct-quantization controls for SRQ-FLY Priority 3.

All methods use one paired CIFAR-100 train/validation stream and one WTA code
cache.  The held-out test cache is forbidden.  Each method runs in an isolated
process so peak CUDA measurements remain method-specific and interrupted runs
can resume without recomputing completed controls.
"""

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

from methods.srq_fly_optimized import DirectInt8GramLearner, SquareRootFLYLearner
from methods.srq_fly_optimized.direct_control import (
    CertifiedDirectInt8GramLearner,
)
from tools import srq_fly_d0 as d0
from tools import srq_fly_priority1_ablation as p1


METHODS = (
    "exact_fly_10000",
    "direct_int8_gram_naive",
    "direct_int8_gram_weyl_repair",
    "sqrt_float16",
    "srq_int8_p2b",
)
TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name",
    "checkpoint_sha256", "feature_dim", "seed", "num_classes", "num_tasks",
    "validation_fraction", "statistics_dtype", "solver_dtype",
    "fly_ridge_lambda", "representation", "storage", "p2b_backend",
    "direct_gram_repair", "gates",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("Priority-3 config keys/schema mismatch")
    if config["seed"] != 2025:
        raise ValueError("Priority-3 development seed must remain 2025")
    if (
        config["dataset"] != "CIFAR-100"
        or config["feature_dim"] <= 0
        or config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
        or not 0 < config["validation_fraction"] < 1
    ):
        raise ValueError("invalid dataset/task/validation protocol")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("Priority-3 controls require float32 statistics and solve")
    if config["fly_ridge_lambda"] <= 0:
        raise ValueError("FLY Ridge lambda must be positive")
    representation = config["representation"]
    if set(representation) != d0.REPRESENTATION_KEYS or min(
        representation[key]
        for key in ("expand_dim", "synaptic_degree", "encode_batch_size", "evaluation_batch_size")
    ) <= 0 or not 0 < representation["coding_level"] <= 1:
        raise ValueError("invalid FLY representation")
    if set(config["storage"]) != d0.STORAGE_KEYS or min(config["storage"].values()) <= 0:
        raise ValueError("invalid compressed storage")
    backend = config["p2b_backend"]
    if backend != {
        "update_backend": "blocked_qr",
        "update_panel_size": 128,
        "first_update_backend": "gram_cholesky",
        "quantization_backend": "streaming",
        "quantization_batch_blocks": 64,
    }:
        raise ValueError("P2B backend identity changed")
    repair = config["direct_gram_repair"]
    if set(repair) != {
        "name", "margin_multiplier", "error_chunk_size",
        "uses_labels_or_accuracy", "adaptive_retry_allowed",
    } or repair["name"] != "weyl_infinity_norm_diagonal_loading":
        raise ValueError("direct-Gram repair identity changed")
    if (
        repair["margin_multiplier"] <= 0
        or repair["error_chunk_size"] <= 0
        or repair["uses_labels_or_accuracy"] is not False
        or repair["adaptive_retry_allowed"] is not False
    ):
        raise ValueError("invalid direct-Gram repair contract")
    gates = config["gates"]
    if set(gates) != {
        "maximum_solver_relative_residual", "practical_equivalence_pp",
        "material_square_root_advantage_pp",
    } or any(float(value) <= 0 for value in gates.values()):
        raise ValueError("invalid Priority-3 gates")
    return config


def _stream_config(config: dict) -> dict:
    """Adapt the locked Priority-3 config to the tested Priority-1 loader."""
    return {
        **config,
        "large_representation": config["representation"],
        # Legacy WTA-cache metadata field required by the shared cache helper;
        # it has no effect on code construction or any Priority-3 learner.
        "raw_ridge_lambda": 0.01,
        "gates": {
            "maximum_solver_relative_residual": config["gates"][
                "maximum_solver_relative_residual"
            ]
        },
    }


def _common_kwargs(config: dict, feature_dim: int, projection, device):
    representation = config["representation"]
    return {
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


def _new_compressed(config: dict, method: str, feature_dim: int, projection, device):
    kwargs = _common_kwargs(config, feature_dim, projection, device)
    if method == "direct_int8_gram_naive":
        return DirectInt8GramLearner(**kwargs)
    if method == "direct_int8_gram_weyl_repair":
        repair = config["direct_gram_repair"]
        return CertifiedDirectInt8GramLearner(
            repair_margin_multiplier=float(repair["margin_multiplier"]),
            repair_error_chunk_size=int(repair["error_chunk_size"]),
            **kwargs,
        )
    mode = "float16" if method == "sqrt_float16" else "int8"
    return SquareRootFLYLearner(
        storage_mode=mode,
        **config["p2b_backend"],
        **kwargs,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate_exact(
    *, config: dict, train: dict, code_cache,
    training_parts: list[torch.Tensor], validation_parts: list[torch.Tensor],
    device: torch.device,
) -> dict:
    """Exact-FLY anchor with update timing aligned to compressed controls."""
    code_indices, code_values, _, projection = code_cache
    representation = config["representation"]
    dimension = int(representation["expand_dim"])
    gram = torch.zeros((dimension, dimension), device=device, dtype=torch.float32)
    cross = torch.zeros((dimension, 0), device=device, dtype=torch.float32)
    counts = torch.zeros(0, device=device, dtype=torch.float32)
    class_ids, stage_accuracy, task_diagnostics = [], [], []
    weights = None
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        task_started = time.perf_counter()
        codes = d0._dense_codes(
            code_indices[indices], code_values[indices], dimension,
            device=device, dtype=torch.float32,
        )
        labels = train["labels"][indices]
        new_ids = sorted(set(class_ids) | set(map(int, labels.tolist())))
        update_started = time.perf_counter()
        cross, counts = d0._expand_cross(cross, counts, class_ids, new_ids)
        class_ids = new_ids
        targets = d0._targets(labels, class_ids, device=device, dtype=torch.float32)
        gram.add_(codes.T @ codes)
        cross.add_(codes.T @ targets)
        counts.add_(targets.sum(0))
        system = gram + config["fly_ridge_lambda"] * torch.eye(
            dimension, device=device, dtype=torch.float32
        )
        weights, residual = d0._solve(system, cross)
        _sync(device)
        update_seconds = time.perf_counter() - update_started
        del codes, system
        accuracy = d0._stage_code_accuracy(
            weights, class_ids, validation_parts, task,
            code_indices, code_values, train["labels"], dimension,
            int(representation["evaluation_batch_size"]),
        )
        stage_accuracy.append(accuracy)
        task_diagnostics.append({
            "task": task + 1,
            "validation_accuracy": accuracy,
            "update_seconds": update_seconds,
            "stage_seconds": time.perf_counter() - task_started,
            "solver_relative_residual": residual,
        })
        print(
            f"TASK method=exact_fly_10000 {task + 1}/{len(training_parts)} "
            f"AA={accuracy:.4f} update={update_seconds:.3f}s",
            flush=True,
        )
    state = (projection, gram, cross, counts, weights)
    return {
        "method": "exact_fly_10000",
        "status": "complete",
        "ridge_lambda": config["fly_ridge_lambda"],
        "validation_average_accuracy": sum(stage_accuracy) / len(stage_accuracy),
        "stage_accuracy": stage_accuracy,
        "persistent_state_bytes": sum(d0._state_bytes(value) for value in state),
        "maximum_solver_relative_residual": max(
            row["solver_relative_residual"] for row in task_diagnostics
        ),
        "total_update_seconds": sum(
            row["update_seconds"] for row in task_diagnostics
        ),
        "analytic_and_validation_seconds": time.perf_counter() - started,
        "task_diagnostics": task_diagnostics,
        "uses_test_set": False,
        "exemplar_free": True,
    }


def _evaluate_compressed(
    *, method: str, config: dict, train: dict, code_cache,
    training_parts: list[torch.Tensor], validation_parts: list[torch.Tensor],
    device: torch.device,
) -> dict:
    code_indices, code_values, _, projection = code_cache
    learner = _new_compressed(
        config, method, int(train["features"].shape[1]), projection, device
    )
    stage_accuracy, task_diagnostics = [], []
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        task_started = time.perf_counter()
        codes = d0._dense_codes(
            code_indices[indices], code_values[indices], learner.expand_dim,
            device=device, dtype=learner.statistics_dtype,
        )
        update_started = time.perf_counter()
        try:
            if method in {"sqrt_float16", "srq_int8_p2b"}:
                learner.update_codes_consuming(codes, train["labels"][indices])
            else:
                learner.update_codes(codes, train["labels"][indices])
        except RuntimeError as error:
            expected = (
                method == "direct_int8_gram_naive"
                and str(error)
                == "compressed Ridge system is not numerically positive definite"
            )
            if not expected:
                raise
            _sync(device)
            print(
                f"NUMERICAL_FAILURE method={method} task={task + 1} reason={error}",
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
                    row["update_seconds"] for row in task_diagnostics
                ) + time.perf_counter() - update_started,
                "task_diagnostics": task_diagnostics,
            }
        _sync(device)
        update_seconds = time.perf_counter() - update_started
        del codes
        accuracy = d0._stage_code_accuracy(
            learner.weights, learner.class_ids, validation_parts, task,
            code_indices, code_values, train["labels"], learner.expand_dim,
            int(config["representation"]["evaluation_batch_size"]),
        )
        stage_accuracy.append(accuracy)
        row = {
            "task": task + 1,
            "validation_accuracy": accuracy,
            "update_seconds": update_seconds,
            "stage_seconds": time.perf_counter() - task_started,
            "persistent_state_bytes": learner.persistent_state_bytes(),
            "solver_relative_residual": float(
                learner.diagnostics["solver_relative_residual"]
            ),
        }
        if method == "direct_int8_gram_weyl_repair":
            row.update({
                key: float(learner.diagnostics[key])
                for key in (
                    "local_quantization_error_infinity_bound",
                    "certified_gram_lower_bound", "diagonal_loading",
                    "effective_ridge_lambda",
                    "certified_system_eigenvalue_floor",
                    "relative_local_storage_error",
                )
            })
        task_diagnostics.append(row)
        suffix = (
            f" loading={row['diagonal_loading']:.6g}"
            if "diagonal_loading" in row else ""
        )
        print(
            f"TASK method={method} {task + 1}/{len(training_parts)} "
            f"AA={accuracy:.4f} update={update_seconds:.3f}s{suffix}",
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
            row["solver_relative_residual"] for row in task_diagnostics
        ),
        "total_update_seconds": sum(
            row["update_seconds"] for row in task_diagnostics
        ),
        "analytic_and_validation_seconds": time.perf_counter() - started,
        "task_diagnostics": task_diagnostics,
    }


def _source_identity() -> dict[str, str]:
    return {
        "runner": _sha256(Path(__file__).resolve()),
        "optimized_learner": _sha256(ROOT / "methods/srq_fly_optimized/learner.py"),
        "optimized_storage": _sha256(ROOT / "methods/srq_fly_optimized/storage.py"),
        "direct_control": _sha256(
            ROOT / "methods/srq_fly_optimized/direct_control.py"
        ),
        "stream_loader": _sha256(ROOT / "tools/srq_fly_priority1_ablation.py"),
        "exact_control": _sha256(ROOT / "tools/srq_fly_d0.py"),
    }


def run_worker(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    feature_cache = Path(args.feature_cache_dir).resolve()
    stream_config = _stream_config(config)
    train, class_order, training_parts, validation_parts, code_cache = p1._load_stream(
        config=stream_config,
        feature_cache_dir=feature_cache,
        code_cache_dir=Path(args.code_cache_dir).resolve(),
        representation=config["representation"],
        device_name=args.device,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if args.method == "exact_fly_10000":
        result = _evaluate_exact(
            config=config, train=train, code_cache=code_cache,
            training_parts=training_parts,
            validation_parts=validation_parts,
            device=device,
        )
    else:
        result = _evaluate_compressed(
            method=args.method, config=config, train=train, code_cache=code_cache,
            training_parts=training_parts, validation_parts=validation_parts,
            device=device,
        )
    result.update(
        class_order=class_order,
        config_sha256=_sha256(config_path),
        source_identity=_source_identity(),
        peak_cuda_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        ),
        peak_cuda_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda" else None
        ),
        peak_memory_scope=(
            "learner construction, analytic update, and validation; "
            "frozen feature extraction and WTA-cache construction excluded"
        ),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _interpret(config: dict, exact: dict, repaired: dict, srq: dict) -> str:
    difference = srq["validation_average_accuracy"] - repaired[
        "validation_average_accuracy"
    ]
    if abs(difference) <= config["gates"]["practical_equivalence_pp"]:
        return "LOW_BITS_SUFFICIENT_AFTER_CERTIFIED_REPAIR_ON_THIS_STREAM"
    if difference >= config["gates"]["material_square_root_advantage_pp"]:
        return "SQUARE_ROOT_STRUCTURE_HAS_MATERIAL_ACCURACY_ADVANTAGE"
    if difference > 0:
        return "SQUARE_ROOT_STRUCTURE_HAS_MODEST_ACCURACY_ADVANTAGE"
    if repaired["validation_average_accuracy"] > exact["validation_average_accuracy"]:
        return "REPAIRED_DIRECT_CONTROL_EXCEEDS_EXACT_REFERENCE_ON_DEVELOPMENT_SPLIT"
    return "REPAIRED_DIRECT_CONTROL_OUTPERFORMS_SQUARE_ROOT_ON_THIS_STREAM"


def run_driver(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    feature_cache = Path(args.feature_cache_dir).resolve()
    if (feature_cache / "test.pt").exists():
        raise RuntimeError("held-out test.pt must remain hidden")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_source = _source_identity()
    results = []
    for method in METHODS:
        output = output_dir / f"{method}.json"
        if output.is_file():
            restored = json.loads(output.read_text(encoding="utf-8"))
            status_ok = restored.get("status") == "complete" or (
                method == "direct_int8_gram_naive"
                and restored.get("status") == "numerical_failure"
            )
            if (
                status_ok
                and restored.get("method") == method
                and restored.get("uses_test_set") is False
                and restored.get("config_sha256") == _sha256(config_path)
                and restored.get("source_identity") == expected_source
            ):
                results.append(restored)
                print(f"RESUME method={method}", flush=True)
                continue
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()), "worker",
            "--config", str(config_path),
            "--feature-cache-dir", str(feature_cache),
            "--code-cache-dir", str(Path(args.code_cache_dir).resolve()),
            "--method", method, "--output", str(output), "--device", args.device,
        ]
        print(f"START method={method}", flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode:
            raise RuntimeError(f"Priority-3 worker failed: {method}")
        results.append(json.loads(output.read_text(encoding="utf-8")))
        print(f"DONE method={method}", flush=True)

    by_method = {row["method"]: row for row in results}
    exact = by_method["exact_fly_10000"]
    naive = by_method["direct_int8_gram_naive"]
    repaired = by_method["direct_int8_gram_weyl_repair"]
    float16 = by_method["sqrt_float16"]
    srq = by_method["srq_int8_p2b"]
    completed = [exact, repaired, float16, srq]
    repair_rows = repaired.get("task_diagnostics", [])
    gates = {
        "all_required_controls_accounted_for": len(results) == len(METHODS),
        "all_non_naive_methods_complete": all(
            row["status"] == "complete" for row in completed
        ),
        "naive_direct_outcome_recorded": naive["status"]
        in {"complete", "numerical_failure"},
        "heldout_test_remained_hidden": not (feature_cache / "test.pt").exists(),
        "all_complete_methods_numerically_stable": max(
            row["maximum_solver_relative_residual"] for row in completed
        ) <= config["gates"]["maximum_solver_relative_residual"],
        "repair_certificate_present_every_task": len(repair_rows)
        == config["num_tasks"]
        and all(
            row["certified_system_eigenvalue_floor"] > 0
            and row["diagonal_loading"] >= 0
            for row in repair_rows
        ),
        "repair_did_not_use_accuracy_or_retry": config["direct_gram_repair"][
            "uses_labels_or_accuracy"
        ] is False
        and config["direct_gram_repair"]["adaptive_retry_allowed"] is False,
    }
    deltas = {
        "srq_minus_repaired_direct_aia_pp": srq["validation_average_accuracy"]
        - repaired["validation_average_accuracy"],
        "float16_sqrt_minus_repaired_direct_aia_pp": float16[
            "validation_average_accuracy"
        ] - repaired["validation_average_accuracy"],
        "repaired_direct_minus_exact_aia_pp": repaired[
            "validation_average_accuracy"
        ] - exact["validation_average_accuracy"],
    }
    summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "COMPLETE_REVIEW_PRIORITY3" if all(gates.values()) else "STOP_PRIORITY3",
        "uses_test_set": False,
        "held_out_test_authorized": False,
        "config_sha256": _sha256(config_path),
        "source_identity": expected_source,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        "methodological_gates": gates,
        "accuracy_deltas": deltas,
        "interpretation": _interpret(config, exact, repaired, srq),
        "repair_observations": {
            "maximum_diagonal_loading": max(
                row["diagonal_loading"] for row in repair_rows
            ),
            "final_diagonal_loading": repair_rows[-1]["diagonal_loading"],
            "final_effective_ridge_lambda": repair_rows[-1][
                "effective_ridge_lambda"
            ],
            "maximum_loading_to_base_ridge_ratio": max(
                row["diagonal_loading"] / config["fly_ridge_lambda"]
                for row in repair_rows
            ),
        },
        "results": results,
    }
    (output_dir / "priority3_results.json").write_text(
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
    driver.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
