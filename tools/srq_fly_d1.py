"""Locked 20-task train-only drift study for SRQ-FLY.

There is deliberately no held-out evaluation mode. Paired metrics are reduced
online; no per-sample validation prediction is persisted.
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

from methods.srq_fly import SquareRootFLYLearner
from tools import srq_fly_d0 as d0
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.tail_fly_phasea import (
    _dense_codes, _expand_cross, _git_provenance, _load_unit, _save_unit,
    _state_bytes, _targets, _unit_path,
)
from tools.twa_fly_pilot import (
    _prepare_code_cache, _sequence_sha256, _sha256_bytes, _sha256_file,
    _tensor_content_sha256,
)


GATE_KEYS = {
    "maximum_solver_relative_residual",
    "maximum_average_gap_to_exact_fly_pp",
    "maximum_final_gap_to_exact_fly_pp",
    "maximum_state_fraction_of_exact_fly",
    "minimum_prediction_agreement",
    "minimum_gain_over_direct_int8_pp",
    "maximum_float16_gap_to_exact_fly_pp",
}


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != d0.TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("config keys/schema mismatch")
    if config["seed"] != 2025:
        raise ValueError("new SRQ-FLY protocols require repository seed 2025")
    if (
        config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
        or config["diagnostic_tasks"] != config["num_tasks"]
    ):
        raise ValueError("D1 must cover the complete valid task stream")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    if config["statistics_dtype"] not in {"float32", "float64"} or config["solver_dtype"] not in {"float32", "float64"}:
        raise ValueError("invalid dtype")
    if config["ridge_lambda"] <= 0 or config["raw_ridge_lambda"] <= 0:
        raise ValueError("Ridge parameters must be positive")
    for name in ("large_representation", "compact_representation"):
        representation = config[name]
        if set(representation) != d0.REPRESENTATION_KEYS:
            raise ValueError(f"{name} keys mismatch")
        if min(representation[key] for key in (
            "expand_dim", "synaptic_degree", "encode_batch_size", "evaluation_batch_size"
        )) <= 0 or not 0 < representation["coding_level"] <= 1:
            raise ValueError(f"invalid {name}")
    if config["large_representation"]["expand_dim"] <= config["compact_representation"]["expand_dim"]:
        raise ValueError("large representation must exceed compact representation")
    if set(config["storage"]) != d0.STORAGE_KEYS or min(config["storage"].values()) <= 0:
        raise ValueError("invalid storage configuration")
    gates = config["gates"]
    if set(gates) != GATE_KEYS:
        raise ValueError("D1 gate keys mismatch")
    if (
        gates["maximum_solver_relative_residual"] <= 0
        or gates["maximum_average_gap_to_exact_fly_pp"] < 0
        or gates["maximum_final_gap_to_exact_fly_pp"] < 0
        or not 0 < gates["maximum_state_fraction_of_exact_fly"] <= 1
        or not 0 <= gates["minimum_prediction_agreement"] <= 1
        or gates["minimum_gain_over_direct_int8_pp"] < 0
        or gates["maximum_float16_gap_to_exact_fly_pp"] < 0
    ):
        raise ValueError("invalid D1 gates")
    return config


def _paired_stage_metrics(
    *, exact_weights: torch.Tensor, approximate_weights: torch.Tensor,
    class_ids: list[int], validation_parts: list[torch.Tensor], task: int,
    code_indices: torch.Tensor, code_values: torch.Tensor, labels: torch.Tensor,
    dimension: int, batch_size: int,
) -> dict[str, float]:
    exact_task_accuracy, approximate_task_accuracy, task_agreement = [], [], []
    squared_difference = 0.0
    squared_exact = 0.0
    for previous in range(task + 1):
        indices = validation_parts[previous]
        exact_correct = approximate_correct = agreements = rows = 0
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            codes = _dense_codes(
                code_indices[selected], code_values[selected], dimension,
                device=exact_weights.device, dtype=exact_weights.dtype,
            )
            exact_logits = codes @ exact_weights
            approximate_logits = codes.to(approximate_weights.dtype) @ approximate_weights
            exact_columns = exact_logits.argmax(1)
            approximate_columns = approximate_logits.argmax(1)
            truth = labels[selected].cpu()
            exact_predictions = torch.tensor(
                [class_ids[column] for column in exact_columns.detach().cpu().tolist()]
            )
            approximate_predictions = torch.tensor(
                [class_ids[column] for column in approximate_columns.detach().cpu().tolist()]
            )
            exact_correct += int((exact_predictions == truth).sum())
            approximate_correct += int((approximate_predictions == truth).sum())
            agreements += int((exact_predictions == approximate_predictions).sum())
            rows += len(selected)
            difference = approximate_logits.to(torch.float64) - exact_logits.to(torch.float64)
            squared_difference += float((difference * difference).sum().item())
            exact64 = exact_logits.to(torch.float64)
            squared_exact += float((exact64 * exact64).sum().item())
        exact_task_accuracy.append(100.0 * exact_correct / max(rows, 1))
        approximate_task_accuracy.append(100.0 * approximate_correct / max(rows, 1))
        task_agreement.append(agreements / max(rows, 1))
    return {
        "exact_accuracy": sum(exact_task_accuracy) / len(exact_task_accuracy),
        "approximate_accuracy": sum(approximate_task_accuracy) / len(approximate_task_accuracy),
        "prediction_agreement": sum(task_agreement) / len(task_agreement),
        "relative_logit_frobenius_error": (squared_difference / max(squared_exact, 1.0)) ** 0.5,
    }


def _evaluate_paired_exact_srq(
    *, config, train, code_indices, code_values, projection,
    training_parts, validation_parts, device,
) -> dict:
    representation = config["large_representation"]
    dtype = d0._dtype(config["statistics_dtype"])
    dimension = int(representation["expand_dim"])
    gram = torch.zeros((dimension, dimension), device=device, dtype=dtype)
    cross = torch.zeros((dimension, 0), device=device, dtype=dtype)
    counts = torch.zeros(0, device=device, dtype=dtype)
    class_ids: list[int] = []
    learner = SquareRootFLYLearner(
        storage_mode="int8", feature_dim=int(train["features"].shape[1]),
        expand_dim=dimension, synaptic_degree=int(representation["synaptic_degree"]),
        coding_level=float(representation["coding_level"]),
        ridge_lambda=float(config["ridge_lambda"]),
        block_size=int(config["storage"]["block_size"]),
        group_size=int(config["storage"]["group_size"]), seed=int(config["seed"]),
        device=device, statistics_dtype=dtype, solver_dtype=d0._dtype(config["solver_dtype"]),
        projection=projection,
    )
    exact_accuracy, srq_accuracy, agreements, logit_errors = [], [], [], []
    exact_residuals, srq_residuals, diagnostics = [], [], []
    exact_weights = None
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        task_started = time.perf_counter()
        codes = _dense_codes(
            code_indices[indices], code_values[indices], dimension,
            device=device, dtype=dtype,
        )
        task_labels = train["labels"][indices]
        new_ids = sorted(set(class_ids) | set(map(int, task_labels.tolist())))
        cross, counts = _expand_cross(cross, counts, class_ids, new_ids)
        class_ids = new_ids
        targets = _targets(task_labels, class_ids, device=device, dtype=dtype)
        gram += codes.T @ codes
        cross += codes.T @ targets
        counts += targets.sum(0)
        system = gram + float(config["ridge_lambda"]) * torch.eye(
            dimension, device=device, dtype=dtype
        )
        exact_weights, exact_residual = d0._solve(system, cross)
        learner.update_codes(codes, task_labels)
        del codes, system
        if learner.class_ids != class_ids:
            raise RuntimeError("paired classifiers have inconsistent class mappings")
        metrics = _paired_stage_metrics(
            exact_weights=exact_weights, approximate_weights=learner.weights,
            class_ids=class_ids, validation_parts=validation_parts, task=task,
            code_indices=code_indices, code_values=code_values, labels=train["labels"],
            dimension=dimension, batch_size=int(representation["evaluation_batch_size"]),
        )
        exact_accuracy.append(metrics["exact_accuracy"])
        srq_accuracy.append(metrics["approximate_accuracy"])
        agreements.append(metrics["prediction_agreement"])
        logit_errors.append(metrics["relative_logit_frobenius_error"])
        exact_residuals.append(exact_residual)
        srq_residuals.append(learner.diagnostics["solver_relative_residual"])
        diagnostic = {
            "task": task + 1, **metrics,
            "accuracy_gap_pp": metrics["exact_accuracy"] - metrics["approximate_accuracy"],
            "exact_solver_relative_residual": exact_residual,
            "srq_solver_relative_residual": learner.diagnostics["solver_relative_residual"],
            "srq_persistent_state_bytes": learner.persistent_state_bytes(),
            "seconds": time.perf_counter() - task_started,
        }
        diagnostics.append(diagnostic)
        print(
            f"TASK paired_exact_srq {task+1}/{len(training_parts)} "
            f"exact={metrics['exact_accuracy']:.4f} srq={metrics['approximate_accuracy']:.4f} "
            f"agree={100*metrics['prediction_agreement']:.3f}% "
            f"logit_err={metrics['relative_logit_frobenius_error']:.3e}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    exact_state = sum(
        _state_bytes(value) for value in (projection, gram, cross, counts, exact_weights)
    )
    seconds = time.perf_counter() - started
    return {
        "status": "complete", "uses_test_set": False,
        "paired_diagnostics": diagnostics,
        "minimum_prediction_agreement": min(agreements),
        "maximum_relative_logit_frobenius_error": max(logit_errors),
        "exact": {
            "method": "exact_fly_10000", "status": "complete",
            "ridge_lambda": config["ridge_lambda"],
            "validation_average_accuracy": sum(exact_accuracy) / len(exact_accuracy),
            "stage_accuracy": exact_accuracy,
            "persistent_state_bytes": exact_state,
            "maximum_solver_relative_residual": max(exact_residuals),
            "seconds": seconds, "uses_test_set": False, "exemplar_free": True,
        },
        "srq": {
            "method": "srq_int8", "status": "complete",
            "ridge_lambda": config["ridge_lambda"],
            "validation_average_accuracy": sum(srq_accuracy) / len(srq_accuracy),
            "stage_accuracy": srq_accuracy,
            "persistent_state_bytes": learner.persistent_state_bytes(),
            "maximum_solver_relative_residual": max(srq_residuals),
            "minimum_prediction_agreement": min(agreements),
            "maximum_relative_logit_frobenius_error": max(logit_errors),
            "task_diagnostics": diagnostics, "seconds": seconds,
            "uses_test_set": False, "exemplar_free": True,
        },
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    if args.require_test_hidden and (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("held-out test.pt is visible")
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
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
            config=d0._cache_config(config, representation), device=args.device,
        )
    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"], task_indices, config["seed"], config["validation_fraction"]
    )
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
        "d0_runner_source_sha256": _sha256_file(ROOT / "tools/srq_fly_d0.py"),
        "learner_source_sha256": _sha256_file(ROOT / "methods/srq_fly/learner.py"),
        "storage_source_sha256": _sha256_file(ROOT / "methods/srq_fly/storage.py"),
    }
    context_sha256 = _sha256_bytes(json.dumps(context, sort_keys=True).encode())
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    large, compact = caches["large"], caches["compact"]

    paired_path = _unit_path(output_dir, "paired_exact_srq")
    paired = _load_unit(paired_path, context_sha256)
    if paired is None:
        print("START paired_exact_srq", flush=True)
        paired = _save_unit(
            paired_path, context_sha256,
            _evaluate_paired_exact_srq(
                config=config, train=train, code_indices=large[0], code_values=large[1],
                projection=large[3], training_parts=training_parts,
                validation_parts=validation_parts, device=device,
            ),
        )
        print("DONE paired_exact_srq status=complete", flush=True)
    results = [paired["exact"], paired["srq"]]

    def unit(name, evaluator):
        path = _unit_path(output_dir, name)
        result = _load_unit(path, context_sha256)
        if result is None:
            print(f"START {name}", flush=True)
            result = _save_unit(path, context_sha256, evaluator())
            print(f"DONE {name} status={result['status']}", flush=True)
        results.append(result)

    unit("exact_fly_4096", lambda: d0._evaluate_exact(
        name="exact_fly_4096", config=config,
        representation=config["compact_representation"], train=train,
        code_indices=compact[0], code_values=compact[1], projection=compact[3],
        training_parts=training_parts, validation_parts=validation_parts, device=device,
    ))
    unit("raw_ridge", lambda: d0._evaluate_raw(
        config=config, train=train, training_parts=training_parts,
        validation_parts=validation_parts, device=device,
    ))
    for method in ("direct_int8_gram", "sqrt_float16"):
        unit(method, lambda method=method: d0._evaluate_learner(
            method=method, config=config, train=train, code_indices=large[0],
            code_values=large[1], projection=large[3], training_parts=training_parts,
            validation_parts=validation_parts, device=device,
        ))

    by_name = {result["method"]: result for result in results}
    exact = by_name["exact_fly_10000"]
    srq = by_name["srq_int8"]
    compact_result = by_name["exact_fly_4096"]
    direct = by_name["direct_int8_gram"]
    float16 = by_name["sqrt_float16"]
    complete = len(results) == 6 and all(result["status"] == "complete" for result in results)
    gates_config = config["gates"]
    gates = {
        "all_units_complete": complete,
        "heldout_test_remained_hidden": not (feature_cache_dir / "test.pt").exists(),
        "numerical_stability": complete and max(
            result["maximum_solver_relative_residual"] for result in results
        ) <= gates_config["maximum_solver_relative_residual"],
        "average_accuracy_within_gate": complete and exact["validation_average_accuracy"] - srq["validation_average_accuracy"] <= gates_config["maximum_average_gap_to_exact_fly_pp"],
        "final_accuracy_within_gate": complete and exact["stage_accuracy"][-1] - srq["stage_accuracy"][-1] <= gates_config["maximum_final_gap_to_exact_fly_pp"],
        "prediction_agreement_within_gate": complete and paired["minimum_prediction_agreement"] >= gates_config["minimum_prediction_agreement"],
        "state_within_gate": complete and srq["persistent_state_bytes"] / exact["persistent_state_bytes"] <= gates_config["maximum_state_fraction_of_exact_fly"],
        "square_root_beats_direct_int8": complete and srq["validation_average_accuracy"] - direct["validation_average_accuracy"] >= gates_config["minimum_gain_over_direct_int8_pp"],
        "float16_tracks_exact": complete and exact["validation_average_accuracy"] - float16["validation_average_accuracy"] <= gates_config["maximum_float16_gap_to_exact_fly_pp"],
        "not_pareto_dominated_by_exact_fly_4096": complete and not (
            compact_result["validation_average_accuracy"] >= srq["validation_average_accuracy"]
            and compact_result["persistent_state_bytes"] <= srq["persistent_state_bytes"]
            and (compact_result["validation_average_accuracy"] > srq["validation_average_accuracy"]
                 or compact_result["persistent_state_bytes"] < srq["persistent_state_bytes"])
        ),
    }
    decision = "PASS_REVIEW_D1" if all(gates.values()) else "STOP_SRQ_FLY_D1"
    payload = {
        "schema_version": 1, "study_id": config["study_id"], "status": decision,
        "uses_test_set": False, "held_out_test_authorized": False,
        "diagnostic_tasks": config["diagnostic_tasks"], "provenance": context,
        "class_order": class_order, "paired_diagnostics": paired["paired_diagnostics"],
        "results": results, "gates": gates,
    }
    path = output_dir / "d1_results.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(f"SRQ-FLY D1 {decision}", flush=True)
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
