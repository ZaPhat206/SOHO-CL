"""Locked train-only feasibility runner for CertiFLY Q1."""

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

from methods.certifly import CertiFLYLearner
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.tail_fly_phasea import (
    _dense_codes,
    _evaluate_exact_fly,
    _evaluate_raw,
    _git_provenance,
    _load_unit,
    _save_unit,
    _stage_code_accuracy,
    _unit_path,
)
from tools.twa_fly_pilot import (
    _prepare_code_cache,
    _sequence_sha256,
    _sha256_bytes,
    _sha256_file,
    _tensor_content_sha256,
)


SCHEMA_VERSION = 1
TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name",
    "checkpoint_sha256", "seed", "num_classes", "num_tasks",
    "validation_fraction", "statistics_dtype", "solver_dtype",
    "representation", "certifly_candidates", "raw_ridge_lambdas",
    "fly_control", "gates",
}
REPRESENTATION_KEYS = {
    "expand_dim", "synaptic_degree", "coding_level", "encode_batch_size",
    "evaluation_batch_size",
}
CANDIDATE_KEYS = {"name", "block_size", "error_fraction", "max_bits"}
FLY_KEYS = {"ridge_lower", "ridge_upper", "statistics_dtype"}
GATE_KEYS = {
    "maximum_solver_relative_residual", "maximum_gap_to_exact_fly_pp",
    "maximum_state_fraction_of_exact_fly",
}


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("config keys/schema mismatch")
    if config["seed"] != 2025:
        raise ValueError("new CertiFLY protocols require repository seed 2025")
    if (
        config["num_classes"] <= 1 or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
    ):
        raise ValueError("invalid class/task count")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    if config["statistics_dtype"] not in {"float32", "float64"}:
        raise ValueError("invalid statistics dtype")
    if config["solver_dtype"] not in {"float32", "float64"}:
        raise ValueError("invalid solver dtype")
    if set(config["representation"]) != REPRESENTATION_KEYS:
        raise ValueError("representation keys mismatch")
    representation = config["representation"]
    if min(
        representation["expand_dim"], representation["synaptic_degree"],
        representation["encode_batch_size"], representation["evaluation_batch_size"],
    ) <= 0 or not 0 < representation["coding_level"] <= 1:
        raise ValueError("invalid representation configuration")
    candidates = config["certifly_candidates"]
    if not candidates or len({item["name"] for item in candidates}) != len(candidates):
        raise ValueError("CertiFLY candidate names must be non-empty and unique")
    for candidate in candidates:
        if set(candidate) != CANDIDATE_KEYS:
            raise ValueError("CertiFLY candidate keys mismatch")
        if candidate["block_size"] <= 0 or candidate["max_bits"] not in {8, 16}:
            raise ValueError("invalid CertiFLY storage candidate")
        if not 0 < candidate["error_fraction"] < 1:
            raise ValueError("invalid certificate fraction")
    if not config["raw_ridge_lambdas"] or any(
        value <= 0 for value in config["raw_ridge_lambdas"]
    ):
        raise ValueError("raw Ridge candidates must be positive")
    if set(config["fly_control"]) != FLY_KEYS:
        raise ValueError("exact-FLY control keys mismatch")
    fly = config["fly_control"]
    if fly["ridge_lower"] >= fly["ridge_upper"] or fly["statistics_dtype"] not in {
        "float32", "float64",
    }:
        raise ValueError("invalid exact-FLY control")
    if set(config["gates"]) != GATE_KEYS:
        raise ValueError("gate keys mismatch")
    gates = config["gates"]
    if (
        gates["maximum_solver_relative_residual"] <= 0
        or gates["maximum_gap_to_exact_fly_pp"] < 0
        or not 0 < gates["maximum_state_fraction_of_exact_fly"] <= 1
    ):
        raise ValueError("invalid Q1 gates")
    return config


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _cache_config(config: dict) -> dict:
    return {
        "seed": config["seed"],
        "num_classes": config["num_classes"],
        "representation": dict(config["representation"]),
        "statistics_dtype": "float32",
        "raw_ridge_lambda": 1.0,
        "solver_tolerance": 1e-5,
        "solver_max_iterations": 100,
    }


def _control_config(config: dict) -> dict:
    """Minimal view consumed by audited exact-FLY/raw-Ridge evaluators."""
    return {
        "statistics_dtype": config["statistics_dtype"],
        "representation": config["representation"],
        "search": {"raw_ridge_lambdas": config["raw_ridge_lambdas"]},
        "fly_control": config["fly_control"],
    }


def _new_learner(config, candidate, feature_dim, projection, device):
    representation = config["representation"]
    return CertiFLYLearner(
        feature_dim=feature_dim,
        expand_dim=int(representation["expand_dim"]),
        synaptic_degree=int(representation["synaptic_degree"]),
        coding_level=float(representation["coding_level"]),
        block_size=int(candidate["block_size"]),
        error_fraction=float(candidate["error_fraction"]),
        max_bits=int(candidate["max_bits"]),
        ridge_lower=float(config["fly_control"]["ridge_lower"]),
        ridge_upper=float(config["fly_control"]["ridge_upper"]),
        seed=int(config["seed"]),
        device=device,
        statistics_dtype=_dtype(config["statistics_dtype"]),
        solver_dtype=_dtype(config["solver_dtype"]),
        projection=projection,
    )


def _evaluate_certifly(
    *, config, candidate, exact, train, code_indices, code_values, projection,
    training_parts, validation_parts, device,
) -> dict:
    learner = _new_learner(
        config, candidate, int(train["features"].shape[1]), projection, device
    )
    representation = config["representation"]
    stage_accuracy, task_diagnostics = [], []
    started = time.perf_counter()
    try:
        for task, indices in enumerate(training_parts):
            task_started = time.perf_counter()
            codes = _dense_codes(
                code_indices[indices], code_values[indices], learner.expand_dim,
                device=device, dtype=learner.statistics_dtype,
            )
            ridge = float(exact["ridge_schedule"][task])
            learner.update_codes(codes, train["labels"][indices], selected_ridge=ridge)
            del codes
            accuracy = _stage_code_accuracy(
                learner.weights, learner.class_ids, validation_parts, task,
                code_indices, code_values, train["labels"], learner.expand_dim,
                int(representation["evaluation_batch_size"]),
            )
            stage_accuracy.append(accuracy)
            diagnostic = {
                "task": task + 1,
                "ridge_lambda": ridge,
                "validation_accuracy": accuracy,
                "persistent_state_bytes": learner.persistent_state_bytes(),
                "gram_error_bound": learner.gram.error_bound,
                "gram_error_fraction_of_ridge": learner.gram.error_bound / ridge,
                "int8_blocks": learner.diagnostics["int8_blocks"],
                "int16_blocks": learner.diagnostics["int16_blocks"],
                "solver_relative_residual": learner.diagnostics[
                    "solver_relative_residual"
                ],
                "seconds": time.perf_counter() - task_started,
            }
            task_diagnostics.append(diagnostic)
            print(
                f"TASK {candidate['name']} {task+1}/{len(training_parts)} "
                f"ridge={ridge:g} AA={accuracy:.4f} "
                f"error/ridge={diagnostic['gram_error_fraction_of_ridge']:.3e} "
                f"bits=8:{diagnostic['int8_blocks']} 16:{diagnostic['int16_blocks']} "
                f"state={diagnostic['persistent_state_bytes']}B "
                f"elapsed={(time.perf_counter()-started)/60:.1f}m",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except (RuntimeError, torch.linalg.LinAlgError) as error:
        return {
            "method": "certifly", "candidate": candidate,
            "status": "certificate_or_solver_failed",
            "failure": f"{type(error).__name__}: {error}",
            "stage_accuracy": stage_accuracy, "task_diagnostics": task_diagnostics,
            "seconds": time.perf_counter() - started,
            "uses_test_set": False, "exemplar_free": True,
        }
    return {
        "method": "certifly", "candidate": candidate, "status": "complete",
        "validation_average_accuracy": sum(stage_accuracy) / len(stage_accuracy),
        "stage_accuracy": stage_accuracy,
        "persistent_state_bytes": learner.persistent_state_bytes(),
        "maximum_solver_relative_residual": max(
            item["solver_relative_residual"] for item in task_diagnostics
        ),
        "maximum_gram_error_fraction_of_ridge": max(
            item["gram_error_fraction_of_ridge"] for item in task_diagnostics
        ),
        "task_diagnostics": task_diagnostics,
        "seconds": time.perf_counter() - started,
        "uses_test_set": False, "exemplar_free": True,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    code_cache_dir = Path(args.code_cache_dir).resolve()
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
    labels = sorted(map(int, torch.unique(train["labels"]).tolist()))
    if labels != list(range(config["num_classes"])):
        raise ValueError("training labels do not match locked classes")

    train_path = feature_cache_dir / "train.pt"
    train_sha256 = _sha256_file(train_path)
    code_indices, code_values, code_metadata, projection = _prepare_code_cache(
        train=train, train_sha256=train_sha256, cache_dir=code_cache_dir,
        config=_cache_config(config), device=args.device,
    )
    projection_sha256 = _tensor_content_sha256(projection)
    if projection_sha256 != code_metadata["projection"]["sha256"]:
        raise RuntimeError("runtime projection does not match WTA cache")
    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"], task_indices, config["seed"], config["validation_fraction"]
    )
    git = _git_provenance()
    context = {
        "config_sha256": _sha256_file(config_path),
        "train_sha256": train_sha256,
        "code_cache_identity_sha256": code_metadata["identity_sha256"],
        "projection_sha256": projection_sha256,
        "training_indices_sha256": _sequence_sha256(training_parts),
        "validation_indices_sha256": _sequence_sha256(validation_parts),
        "runner_git_commit": git["git_commit"], "runner_git_dirty": git["git_dirty"],
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "learner_source_sha256": _sha256_file(ROOT / "methods/certifly/learner.py"),
        "quantization_source_sha256": _sha256_file(ROOT / "methods/certifly/quantization.py"),
        "solver_source_sha256": _sha256_file(ROOT / "methods/certifly/solver.py"),
    }
    context_sha256 = _sha256_bytes(json.dumps(context, sort_keys=True).encode())
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    control = _control_config(config)

    exact_path = _unit_path(output_dir, "matched_exact_fly")
    exact = _load_unit(exact_path, context_sha256)
    if exact is None:
        print("START matched_exact_fly", flush=True)
        exact = _evaluate_exact_fly(
            control, train, code_indices, code_values, projection,
            training_parts, validation_parts, device,
        )
        exact = _save_unit(exact_path, context_sha256, exact)
        print("DONE matched_exact_fly", flush=True)

    raw_path = _unit_path(output_dir, "raw_ridge")
    raw = _load_unit(raw_path, context_sha256)
    if raw is None:
        print("START raw_ridge", flush=True)
        raw = _evaluate_raw(control, train, training_parts, validation_parts, device)
        raw = _save_unit(raw_path, context_sha256, raw)
        print("DONE raw_ridge", flush=True)

    certifly_results = []
    for candidate in config["certifly_candidates"]:
        path = _unit_path(output_dir, candidate["name"])
        result = _load_unit(path, context_sha256)
        if result is None:
            print(f"START {candidate['name']}", flush=True)
            result = _evaluate_certifly(
                config=config, candidate=candidate, exact=exact, train=train,
                code_indices=code_indices, code_values=code_values,
                projection=projection, training_parts=training_parts,
                validation_parts=validation_parts, device=device,
            )
            result = _save_unit(path, context_sha256, result)
            print(f"DONE {candidate['name']} status={result['status']}", flush=True)
        certifly_results.append(result)

    completed = [item for item in certifly_results if item["status"] == "complete"]
    selected = max(
        completed,
        key=lambda item: (
            item["validation_average_accuracy"], -item["persistent_state_bytes"],
            item["candidate"]["name"],
        ),
    ) if completed else None
    selected_raw = max(
        raw["candidates"],
        key=lambda item: (item["validation_average_accuracy"], -item["persistent_state_bytes"]),
    )
    if selected is None:
        gates = {
            "heldout_test_remained_hidden": True, "numerical_stability": False,
            "within_accuracy_gate": False, "within_state_gate": False,
        }
    else:
        gates = {
            "heldout_test_remained_hidden": not (feature_cache_dir / "test.pt").exists(),
            "numerical_stability": selected["maximum_solver_relative_residual"]
            <= config["gates"]["maximum_solver_relative_residual"],
            "within_accuracy_gate": exact["validation_average_accuracy"]
            - selected["validation_average_accuracy"]
            <= config["gates"]["maximum_gap_to_exact_fly_pp"],
            "within_state_gate": selected["persistent_state_bytes"]
            / exact["persistent_state_bytes"]
            <= config["gates"]["maximum_state_fraction_of_exact_fly"],
        }
    decision = "PASS_REVIEW_Q1" if all(gates.values()) else "STOP_CERTIFLY_Q1"
    payload = {
        "schema_version": 1, "study_id": config["study_id"], "status": decision,
        "uses_test_set": False, "held_out_test_authorized": False,
        "provenance": context, "class_order": class_order,
        "matched_exact_fly": exact, "selected_raw_ridge": selected_raw,
        "certifly_candidates": certifly_results,
        "selected_certifly": selected, "gates": gates,
    }
    result_path = output_dir / "q1_results.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    print(f"CERTIFLY Q1 {decision}", flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--code-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-test-hidden", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
