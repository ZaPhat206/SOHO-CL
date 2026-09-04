"""Paired CIFAR train-only robustness test for 10 versus 20 SRQ updates.

Each replicate fixes its class order, per-class train/validation membership,
projection, WTA codes, Ridge coefficient, and learner implementation.  Only
the grouping of the 100 classes changes: 10 classes per update versus 5.  The
runner has no test mode and rejects any feature cache containing ``test.pt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.srq_fly_optimized import SquareRootFLYLearner
from tools import srq_fly_d0 as d0
from tools.experiment_runner import validate_cache
from tools.twa_fly_pilot import (
    _prepare_code_cache, _sequence_sha256, _sha256_bytes, _sha256_file,
    _tensor_content_sha256,
)


TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name",
    "checkpoint_sha256", "feature_dim", "seed", "num_classes",
    "task_schedules", "validation_fraction", "replicates",
    "statistics_dtype", "solver_dtype", "fly_ridge_lambda",
    "hyperparameter_policy", "representation", "storage", "p2b_backend",
    "gates",
}
GATE_KEYS = {
    "maximum_solver_relative_residual", "maximum_aligned_aia_gap_to_exact_pp",
    "maximum_final_gap_to_exact_pp", "maximum_mean_added_frequency_loss_pp",
    "minimum_srq_exact_prediction_agreement",
    "maximum_exact_final_accuracy_schedule_difference_pp",
    "minimum_exact_final_prediction_schedule_agreement",
    "maximum_srq_state_fraction_of_exact",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("Priority-4 config keys/schema mismatch")
    if config["seed"] != 2025:
        raise ValueError("Priority-4 protocol seed must remain 2025")
    if (
        config["dataset"] != "CIFAR-100" or config["feature_dim"] <= 0
        or config["num_classes"] != 100
        or config["task_schedules"] != [10, 20]
        or any(config["num_classes"] % value for value in config["task_schedules"])
        or not 0 < config["validation_fraction"] < 1
        or config["fly_ridge_lambda"] <= 0
    ):
        raise ValueError("invalid Priority-4 dataset/task protocol")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("Priority-4 requires float32 statistics and solve")
    replicates = config["replicates"]
    if len(replicates) != 5 or [row.get("id") for row in replicates] != list(range(1, 6)):
        raise ValueError("Priority-4 requires five preregistered replicates")
    seed_tuples = []
    for row in replicates:
        if set(row) != {"id", "class_order_seed", "projection_seed", "split_seed"}:
            raise ValueError("replicate keys mismatch")
        seed_tuples.append(tuple(int(row[key]) for key in (
            "class_order_seed", "projection_seed", "split_seed"
        )))
    if len(set(seed_tuples)) != len(seed_tuples):
        raise ValueError("replicate seeds must be unique")
    policy = config["hyperparameter_policy"]
    if set(policy) != {
        "selection_source", "reference_artifact_sha256", "selection_json_sha256",
        "retuning_allowed", "accuracy_based_early_stop",
    } or policy["retuning_allowed"] is not False or policy["accuracy_based_early_stop"] is not False:
        raise ValueError("Priority-4 forbids retuning and accuracy early stop")
    representation = config["representation"]
    if set(representation) != d0.REPRESENTATION_KEYS or min(
        representation[key] for key in (
            "expand_dim", "synaptic_degree", "encode_batch_size",
            "evaluation_batch_size",
        )
    ) <= 0 or not 0 < representation["coding_level"] <= 1:
        raise ValueError("invalid representation")
    if set(config["storage"]) != d0.STORAGE_KEYS or min(config["storage"].values()) <= 0:
        raise ValueError("invalid storage")
    if config["p2b_backend"] != {
        "storage_mode": "int8", "update_backend": "blocked_qr",
        "update_panel_size": 128, "first_update_backend": "gram_cholesky",
        "quantization_backend": "streaming", "quantization_batch_blocks": 64,
    }:
        raise ValueError("P2B backend identity changed")
    gates = config["gates"]
    if set(gates) != GATE_KEYS or any(float(value) < 0 for value in gates.values()):
        raise ValueError("invalid Priority-4 gates")
    if gates["maximum_solver_relative_residual"] <= 0:
        raise ValueError("solver tolerance must be positive")
    for name in (
        "minimum_srq_exact_prediction_agreement",
        "minimum_exact_final_prediction_schedule_agreement",
        "maximum_srq_state_fraction_of_exact",
    ):
        if not 0 < gates[name] <= 1:
            raise ValueError(f"invalid fraction gate: {name}")
    return config


def _source_identity() -> dict[str, str]:
    return {
        "runner": _sha256(Path(__file__).resolve()),
        "optimized_learner": _sha256(ROOT / "methods/srq_fly_optimized/learner.py"),
        "optimized_storage": _sha256(ROOT / "methods/srq_fly_optimized/storage.py"),
        "exact_helper": _sha256(ROOT / "tools/srq_fly_d0.py"),
        "code_cache_helper": _sha256(ROOT / "tools/twa_fly_pilot.py"),
    }


def _cache_config(config: dict, projection_seed: int) -> dict:
    return {
        "seed": int(projection_seed), "num_classes": config["num_classes"],
        "representation": dict(config["representation"]),
        "statistics_dtype": config["statistics_dtype"],
        "raw_ridge_lambda": 1.0,
        "solver_tolerance": config["gates"]["maximum_solver_relative_residual"],
        "solver_max_iterations": 100,
    }


def _per_class_split(
    labels: torch.Tensor, class_order: list[int], split_seed: int,
    validation_fraction: float,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Create schedule-independent, deterministic membership per class."""
    training, validation = {}, {}
    for class_id in class_order:
        indices = (labels == class_id).nonzero().flatten()
        if len(indices) < 2:
            raise ValueError(f"class {class_id} has fewer than two training samples")
        generator = torch.Generator().manual_seed(
            int(split_seed) * 1009 + int(class_id)
        )
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        count = max(1, min(len(indices) - 1, int(round(
            len(indices) * validation_fraction
        ))))
        validation[class_id] = shuffled[:count].sort().values
        training[class_id] = shuffled[count:].sort().values
    return training, validation


def _group_parts(
    per_class: dict[int, torch.Tensor], class_order: list[int], num_tasks: int,
) -> list[torch.Tensor]:
    per_task = len(class_order) // num_tasks
    return [
        torch.cat([
            per_class[class_id]
            for class_id in class_order[start:start + per_task]
        ]).sort().values
        for start in range(0, len(class_order), per_task)
    ]


def _predict(
    *, weights: torch.Tensor, class_ids: list[int], parts: list[torch.Tensor],
    task: int, code_indices: torch.Tensor, code_values: torch.Tensor,
    labels: torch.Tensor, dimension: int, batch_size: int, device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    indices = torch.cat(parts[:task + 1]).sort().values
    predictions = []
    classes = torch.tensor(class_ids, dtype=torch.long)
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        codes = d0._dense_codes(
            code_indices[batch], code_values[batch], dimension,
            device=device, dtype=torch.float32,
        )
        columns = (codes @ weights).argmax(1).detach().cpu()
        predictions.append(classes[columns])
        del codes
    predicted = torch.cat(predictions)
    accuracy = float((predicted == labels[indices].cpu()).float().mean() * 100)
    return accuracy, indices, predicted


def _new_srq(config: dict, projection, projection_seed: int, device: torch.device):
    representation = config["representation"]
    return SquareRootFLYLearner(
        feature_dim=config["feature_dim"],
        expand_dim=representation["expand_dim"],
        synaptic_degree=representation["synaptic_degree"],
        coding_level=representation["coding_level"],
        ridge_lambda=config["fly_ridge_lambda"],
        block_size=config["storage"]["block_size"],
        group_size=config["storage"]["group_size"],
        seed=projection_seed, device=device,
        statistics_dtype=torch.float32, solver_dtype=torch.float32,
        projection=projection, **config["p2b_backend"],
    )


def _evaluate_pair(
    *, config: dict, train: dict, code_cache, training_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor], projection_seed: int,
    device: torch.device,
) -> dict:
    code_indices, code_values, _, projection = code_cache
    dimension = config["representation"]["expand_dim"]
    gram = torch.zeros((dimension, dimension), device=device, dtype=torch.float32)
    cross = torch.zeros((dimension, 0), device=device, dtype=torch.float32)
    counts = torch.zeros(0, device=device, dtype=torch.float32)
    class_ids: list[int] = []
    exact_weights = None
    learner = _new_srq(config, projection, projection_seed, device)
    exact_accuracy, srq_accuracy, agreements, diagnostics = [], [], [], []
    exact_update_total = srq_update_total = 0.0
    final_indices = final_exact = final_srq = None
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        codes = d0._dense_codes(
            code_indices[indices], code_values[indices], dimension,
            device=device, dtype=torch.float32,
        )
        labels = train["labels"][indices]
        new_ids = sorted(set(class_ids) | set(map(int, labels.tolist())))
        exact_started = time.perf_counter()
        cross, counts = d0._expand_cross(cross, counts, class_ids, new_ids)
        class_ids = new_ids
        targets = d0._targets(labels, class_ids, device=device, dtype=torch.float32)
        gram.add_(codes.T @ codes)
        cross.add_(codes.T @ targets)
        counts.add_(targets.sum(0))
        system = gram + config["fly_ridge_lambda"] * torch.eye(
            dimension, device=device, dtype=torch.float32
        )
        exact_weights, exact_residual = d0._solve(system, cross)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        exact_seconds = time.perf_counter() - exact_started
        del system

        srq_started = time.perf_counter()
        learner.update_codes_consuming(codes, labels)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        srq_seconds = time.perf_counter() - srq_started
        if learner.class_ids != class_ids:
            raise AssertionError("Exact and SRQ class columns diverged")
        exact_value, exact_indices, exact_predictions = _predict(
            weights=exact_weights, class_ids=class_ids, parts=validation_parts,
            task=task, code_indices=code_indices, code_values=code_values,
            labels=train["labels"], dimension=dimension,
            batch_size=config["representation"]["evaluation_batch_size"],
            device=device,
        )
        srq_value, srq_indices, srq_predictions = _predict(
            weights=learner.weights, class_ids=learner.class_ids,
            parts=validation_parts, task=task, code_indices=code_indices,
            code_values=code_values, labels=train["labels"], dimension=dimension,
            batch_size=config["representation"]["evaluation_batch_size"],
            device=device,
        )
        if not torch.equal(exact_indices, srq_indices):
            raise AssertionError("paired validation indices diverged")
        agreement = float((exact_predictions == srq_predictions).float().mean())
        exact_accuracy.append(exact_value)
        srq_accuracy.append(srq_value)
        agreements.append(agreement)
        exact_update_total += exact_seconds
        srq_update_total += srq_seconds
        diagnostics.append({
            "task": task + 1, "seen_classes": len(class_ids),
            "exact_accuracy": exact_value, "srq_accuracy": srq_value,
            "srq_minus_exact_pp": srq_value - exact_value,
            "prediction_agreement": agreement,
            "exact_solver_relative_residual": exact_residual,
            "srq_solver_relative_residual": float(
                learner.diagnostics["solver_relative_residual"]
            ),
            "exact_update_seconds": exact_seconds,
            "srq_update_seconds": srq_seconds,
        })
        final_indices, final_exact, final_srq = (
            exact_indices, exact_predictions, srq_predictions
        )
        print(
            f"TASK {task + 1}/{len(training_parts)} seen={len(class_ids)} "
            f"Exact={exact_value:.4f} SRQ={srq_value:.4f} "
            f"agree={agreement:.5f}", flush=True,
        )
    exact_state = sum(d0._state_bytes(value) for value in (
        projection, gram, cross, counts, exact_weights
    ))
    learner.assert_exemplar_free_state()
    return {
        "status": "complete", "uses_test_set": False,
        "held_out_test_authorized": False,
        "exact": {
            "stage_accuracy": exact_accuracy,
            "validation_average_accuracy": statistics.fmean(exact_accuracy),
            "final_accuracy": exact_accuracy[-1],
            "persistent_state_bytes": exact_state,
            "maximum_solver_relative_residual": max(
                row["exact_solver_relative_residual"] for row in diagnostics
            ),
            "total_update_seconds": exact_update_total,
        },
        "srq": {
            "stage_accuracy": srq_accuracy,
            "validation_average_accuracy": statistics.fmean(srq_accuracy),
            "final_accuracy": srq_accuracy[-1],
            "persistent_state_bytes": learner.persistent_state_bytes(),
            "maximum_solver_relative_residual": max(
                row["srq_solver_relative_residual"] for row in diagnostics
            ),
            "total_update_seconds": srq_update_total,
        },
        "stage_prediction_agreement": agreements,
        "minimum_prediction_agreement": min(agreements),
        "final_validation_indices_sha256": _tensor_content_sha256(final_indices),
        "final_exact_predictions": final_exact.tolist(),
        "final_srq_predictions": final_srq.tolist(),
        "task_diagnostics": diagnostics,
        "paired_seconds": time.perf_counter() - started,
    }


def run_worker(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    replicate = next(
        (row for row in config["replicates"] if row["id"] == args.replicate), None
    )
    if replicate is None or args.num_tasks not in config["task_schedules"]:
        raise ValueError("worker replicate/task schedule is not preregistered")
    feature_cache = Path(args.feature_cache_dir).resolve()
    if (feature_cache / "test.pt").exists():
        raise RuntimeError("held-out test.pt must remain hidden")
    train, _, metadata = validate_cache(
        feature_cache,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint identity mismatch")
    if tuple(train["features"].shape) != (50000, config["feature_dim"]):
        raise ValueError("CIFAR training cache shape mismatch")
    class_order = random.Random(replicate["class_order_seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    per_class_train, per_class_validation = _per_class_split(
        train["labels"], class_order, replicate["split_seed"],
        config["validation_fraction"],
    )
    training_parts = _group_parts(per_class_train, class_order, args.num_tasks)
    validation_parts = _group_parts(per_class_validation, class_order, args.num_tasks)
    code_cache = _prepare_code_cache(
        train=train, train_sha256=_sha256_file(feature_cache / "train.pt"),
        cache_dir=Path(args.code_cache_root).resolve()
        / f"projection_seed_{replicate['projection_seed']}",
        config=_cache_config(config, replicate["projection_seed"]),
        device=args.device,
    )
    device = torch.device(args.device)
    result = _evaluate_pair(
        config=config, train=train, code_cache=code_cache,
        training_parts=training_parts, validation_parts=validation_parts,
        projection_seed=replicate["projection_seed"], device=device,
    )
    result.update({
        "replicate": replicate, "num_tasks": args.num_tasks,
        "classes_per_task": config["num_classes"] // args.num_tasks,
        "class_order": class_order,
        "class_order_sha256": _sha256_bytes(
            json.dumps(class_order).encode("utf-8")
        ),
        "training_indices_sha256": _sequence_sha256(training_parts),
        "validation_indices_sha256": _sequence_sha256(validation_parts),
        "config_sha256": _sha256(config_path),
        "source_identity": _source_identity(),
    })
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _summary(values: list[float]) -> dict:
    parsed = list(map(float, values))
    mean = statistics.fmean(parsed)
    std = statistics.stdev(parsed) if len(parsed) > 1 else 0.0
    half = 2.776 * std / math.sqrt(5) if len(parsed) == 5 else None
    return {
        "values": parsed, "mean": mean, "sample_std": std,
        "ci95_low": None if half is None else mean - half,
        "ci95_high": None if half is None else mean + half,
    }


def _replicate_comparison(low: dict, high: dict) -> dict:
    if low["replicate"] != high["replicate"]:
        raise ValueError("schedule replicate identities differ")
    if low["final_validation_indices_sha256"] != high["final_validation_indices_sha256"]:
        raise ValueError("10-task and 20-task validation membership differs")
    stride = high["num_tasks"] // low["num_tasks"]
    if stride != 2:
        raise ValueError("Priority-4 requires a 2x update-frequency comparison")
    aligned_high = list(range(stride - 1, high["num_tasks"], stride))
    exact_low = statistics.fmean(low["exact"]["stage_accuracy"])
    srq_low = statistics.fmean(low["srq"]["stage_accuracy"])
    exact_high = statistics.fmean(
        high["exact"]["stage_accuracy"][index] for index in aligned_high
    )
    srq_high = statistics.fmean(
        high["srq"]["stage_accuracy"][index] for index in aligned_high
    )
    low_loss = exact_low - srq_low
    high_loss = exact_high - srq_high
    exact_predictions_low = torch.tensor(low["final_exact_predictions"])
    exact_predictions_high = torch.tensor(high["final_exact_predictions"])
    exact_schedule_agreement = float(
        (exact_predictions_low == exact_predictions_high).float().mean()
    )
    return {
        "replicate": low["replicate"],
        "aligned_exact_aia_10": exact_low,
        "aligned_exact_aia_20": exact_high,
        "aligned_srq_aia_10": srq_low,
        "aligned_srq_aia_20": srq_high,
        "srq_exact_loss_10_pp": low_loss,
        "srq_exact_loss_20_pp": high_loss,
        "added_frequency_loss_pp": high_loss - low_loss,
        "exact_final_accuracy_schedule_difference_pp": abs(
            low["exact"]["final_accuracy"] - high["exact"]["final_accuracy"]
        ),
        "exact_final_prediction_schedule_agreement": exact_schedule_agreement,
        "srq_final_accuracy_10": low["srq"]["final_accuracy"],
        "srq_final_accuracy_20": high["srq"]["final_accuracy"],
        "minimum_srq_exact_prediction_agreement_10": low[
            "minimum_prediction_agreement"
        ],
        "minimum_srq_exact_prediction_agreement_20": high[
            "minimum_prediction_agreement"
        ],
    }


def run_driver(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    feature_cache = Path(args.feature_cache_dir).resolve()
    if (feature_cache / "test.pt").exists():
        raise RuntimeError("held-out test.pt must remain hidden")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_source = _source_identity()
    units = []
    for replicate in config["replicates"]:
        for num_tasks in config["task_schedules"]:
            output = output_dir / f"replicate_{replicate['id']}_tasks_{num_tasks}.json"
            restored = None
            if output.is_file():
                candidate = json.loads(output.read_text(encoding="utf-8"))
                if (
                    candidate.get("status") == "complete"
                    and candidate.get("uses_test_set") is False
                    and candidate.get("replicate") == replicate
                    and candidate.get("num_tasks") == num_tasks
                    and candidate.get("config_sha256") == _sha256(config_path)
                    and candidate.get("source_identity") == expected_source
                ):
                    restored = candidate
            if restored is None:
                command = [
                    sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                    "--config", str(config_path),
                    "--feature-cache-dir", str(feature_cache),
                    "--code-cache-root", str(Path(args.code_cache_root).resolve()),
                    "--replicate", str(replicate["id"]),
                    "--num-tasks", str(num_tasks), "--output", str(output),
                    "--device", args.device,
                ]
                print(
                    f"START replicate={replicate['id']} tasks={num_tasks}", flush=True
                )
                completed = subprocess.run(command, cwd=ROOT)
                if completed.returncode:
                    raise RuntimeError(
                        f"Priority-4 worker failed: replicate={replicate['id']} "
                        f"tasks={num_tasks}"
                    )
                restored = json.loads(output.read_text(encoding="utf-8"))
                print(
                    f"DONE replicate={replicate['id']} tasks={num_tasks}", flush=True
                )
            else:
                print(
                    f"RESUME replicate={replicate['id']} tasks={num_tasks}", flush=True
                )
            units.append(restored)
    indexed = {(row["replicate"]["id"], row["num_tasks"]): row for row in units}
    comparisons = [
        _replicate_comparison(indexed[(replicate["id"], 10)], indexed[(replicate["id"], 20)])
        for replicate in config["replicates"]
    ]
    summaries = {
        key: _summary([row[key] for row in comparisons])
        for key in (
            "aligned_exact_aia_10", "aligned_exact_aia_20",
            "aligned_srq_aia_10", "aligned_srq_aia_20",
            "srq_exact_loss_10_pp", "srq_exact_loss_20_pp",
            "added_frequency_loss_pp", "srq_final_accuracy_10",
            "srq_final_accuracy_20",
        )
    }
    gates_config = config["gates"]
    maximum_residual = max(
        row[method]["maximum_solver_relative_residual"]
        for row in units for method in ("exact", "srq")
    )
    minimum_agreement = min(
        row["minimum_prediction_agreement"] for row in units
    )
    maximum_exact_accuracy_difference = max(
        row["exact_final_accuracy_schedule_difference_pp"] for row in comparisons
    )
    minimum_exact_schedule_agreement = min(
        row["exact_final_prediction_schedule_agreement"] for row in comparisons
    )
    maximum_state_fraction = max(
        row["srq"]["persistent_state_bytes"]
        / row["exact"]["persistent_state_bytes"] for row in units
    )
    gates = {
        "all_units_complete": len(units) == 10 and all(
            row["status"] == "complete" for row in units
        ),
        "heldout_test_remained_hidden": not (feature_cache / "test.pt").exists(),
        "all_units_numerically_stable": maximum_residual
        <= gates_config["maximum_solver_relative_residual"],
        "ten_task_aligned_aia_within_exact_gate": summaries[
            "srq_exact_loss_10_pp"
        ]["mean"] <= gates_config["maximum_aligned_aia_gap_to_exact_pp"],
        "twenty_task_aligned_aia_within_exact_gate": summaries[
            "srq_exact_loss_20_pp"
        ]["mean"] <= gates_config["maximum_aligned_aia_gap_to_exact_pp"],
        "twenty_task_final_within_exact_gate": statistics.fmean(
            indexed[(replicate["id"], 20)]["exact"]["final_accuracy"]
            - indexed[(replicate["id"], 20)]["srq"]["final_accuracy"]
            for replicate in config["replicates"]
        ) <= gates_config["maximum_final_gap_to_exact_pp"],
        "added_frequency_loss_within_gate": summaries[
            "added_frequency_loss_pp"
        ]["mean"] <= gates_config["maximum_mean_added_frequency_loss_pp"],
        "srq_exact_prediction_agreement_within_gate": minimum_agreement
        >= gates_config["minimum_srq_exact_prediction_agreement"],
        "exact_final_accuracy_schedule_invariant": maximum_exact_accuracy_difference
        <= gates_config["maximum_exact_final_accuracy_schedule_difference_pp"],
        "exact_final_predictions_schedule_invariant": minimum_exact_schedule_agreement
        >= gates_config["minimum_exact_final_prediction_schedule_agreement"],
        "srq_state_below_quarter_exact": maximum_state_fraction
        <= gates_config["maximum_srq_state_fraction_of_exact"],
    }
    summary = {
        "schema_version": 1, "study_id": config["study_id"],
        "status": "PASS_PRIORITY4_TASK_FREQUENCY" if all(gates.values())
        else "STOP_PRIORITY4_TASK_FREQUENCY",
        "uses_test_set": False, "held_out_test_authorized": False,
        "scientific_question": (
            "Does doubling the number of decode/update/re-quantize events from "
            "10 to 20 materially increase SRQ error when all paired identities "
            "and aligned seen-class checkpoints are fixed?"
        ),
        "config_sha256": _sha256(config_path),
        "source_identity": expected_source,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        "gates": gates, "observed": {
            "maximum_solver_relative_residual": maximum_residual,
            "minimum_srq_exact_prediction_agreement": minimum_agreement,
            "maximum_exact_final_accuracy_schedule_difference_pp": maximum_exact_accuracy_difference,
            "minimum_exact_final_prediction_schedule_agreement": minimum_exact_schedule_agreement,
            "maximum_srq_state_fraction_of_exact": maximum_state_fraction,
        },
        "summaries": summaries, "replicate_comparisons": comparisons,
        "unit_files": [
            f"replicate_{row['replicate']['id']}_tasks_{row['num_tasks']}.json"
            for row in units
        ],
    }
    (output_dir / "priority4_results.json").write_text(
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
        item.add_argument("--code-cache-root", required=True)
        item.add_argument("--device", default="cpu")
    worker.add_argument("--replicate", type=int, required=True)
    worker.add_argument("--num-tasks", type=int, required=True)
    worker.add_argument("--output", required=True)
    driver.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
