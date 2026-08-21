"""Locked exact-FLY state-matched falsification control for SRQ-FLY."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import srq_fly_d0 as d0
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.tail_fly_phasea import _git_provenance, _load_unit, _save_unit, _unit_path
from tools.twa_fly_pilot import (
    _prepare_code_cache, _sequence_sha256, _sha256_bytes, _sha256_file,
    _tensor_content_sha256,
)


TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name", "feature_dim", "checkpoint_sha256",
    "seed", "num_classes", "num_tasks", "validation_fraction",
    "statistics_dtype", "ridge_lambda", "representation", "reference_d1",
    "expected_exact_state_bytes", "gates",
}
REFERENCE_KEYS = {
    "config_sha256", "runner_git_commit", "train_sha256",
    "training_indices_sha256", "validation_indices_sha256",
    "srq_persistent_state_bytes", "srq_validation_average_accuracy",
    "srq_final_accuracy",
}
GATE_KEYS = {
    "maximum_solver_relative_residual", "maximum_state_mismatch_fraction",
    "minimum_srq_average_gain_pp", "minimum_srq_final_gain_pp",
}


def exact_fly_state_bytes(
    *, feature_dim: int, expand_dim: int, synaptic_degree: int,
    num_classes: int,
) -> int:
    if min(feature_dim, expand_dim, synaptic_degree, num_classes) <= 0:
        raise ValueError("state dimensions must be positive")
    projection = (
        expand_dim * synaptic_degree * 4
        + expand_dim * synaptic_degree * 8
        + (feature_dim + 1) * 8
    )
    gram = expand_dim * expand_dim * 4
    cross_classifier_counts = 2 * expand_dim * num_classes * 4 + num_classes * 4
    return projection + gram + cross_classifier_counts


def _read_config(path: Path) -> dict:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate config key: {key}")
            result[key] = value
        return result

    config = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("config keys/schema mismatch")
    if config["seed"] != 2025:
        raise ValueError("new SRQ-FLY protocols require repository seed 2025")
    if (
        config["feature_dim"] <= 0 or config["num_classes"] <= 1 or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
    ):
        raise ValueError("invalid class/task count")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    if config["statistics_dtype"] != "float32":
        raise ValueError("D2 state accounting requires float32 statistics")
    if config["ridge_lambda"] <= 0:
        raise ValueError("ridge_lambda must be positive")
    representation = config["representation"]
    if set(representation) != d0.REPRESENTATION_KEYS:
        raise ValueError("representation keys mismatch")
    if min(representation[key] for key in (
        "expand_dim", "synaptic_degree", "encode_batch_size", "evaluation_batch_size"
    )) <= 0 or not 0 < representation["coding_level"] <= 1:
        raise ValueError("invalid representation")
    reference = config["reference_d1"]
    if set(reference) != REFERENCE_KEYS:
        raise ValueError("D1 reference keys mismatch")
    if min(
        reference["srq_persistent_state_bytes"],
        reference["srq_validation_average_accuracy"],
        reference["srq_final_accuracy"],
    ) <= 0:
        raise ValueError("invalid D1 reference metrics")
    gates = config["gates"]
    if set(gates) != GATE_KEYS:
        raise ValueError("gate keys mismatch")
    if (
        gates["maximum_solver_relative_residual"] <= 0
        or not 0 <= gates["maximum_state_mismatch_fraction"] < 1
        or gates["minimum_srq_average_gain_pp"] < 0
        or gates["minimum_srq_final_gain_pp"] < 0
    ):
        raise ValueError("invalid gates")
    analytic = exact_fly_state_bytes(
        feature_dim=int(config["feature_dim"]), expand_dim=int(representation["expand_dim"]),
        synaptic_degree=int(representation["synaptic_degree"]),
        num_classes=int(config["num_classes"]),
    )
    if analytic != config["expected_exact_state_bytes"]:
        raise ValueError("expected exact state does not match analytic accounting")
    lower = exact_fly_state_bytes(
        feature_dim=int(config["feature_dim"]), expand_dim=int(representation["expand_dim"]) - 1,
        synaptic_degree=int(representation["synaptic_degree"]),
        num_classes=int(config["num_classes"]),
    )
    upper = exact_fly_state_bytes(
        feature_dim=int(config["feature_dim"]), expand_dim=int(representation["expand_dim"]) + 1,
        synaptic_degree=int(representation["synaptic_degree"]),
        num_classes=int(config["num_classes"]),
    )
    target = reference["srq_persistent_state_bytes"]
    if analytic > target or abs(target - analytic) > min(abs(target - lower), abs(target - upper)):
        raise ValueError("representation is not the closest non-exceeding state match")
    return config


def _cache_config(config: dict) -> dict:
    return {
        "seed": config["seed"], "num_classes": config["num_classes"],
        "representation": dict(config["representation"]),
        "statistics_dtype": config["statistics_dtype"],
        "raw_ridge_lambda": 1.0, "solver_tolerance": 1e-5,
        "solver_max_iterations": 100,
    }


def _load_and_verify_d1(
    *, path: Path, config: dict, train_sha256: str,
    class_order: list[int], training_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "STOP_SRQ_FLY_D1":
        raise ValueError("D1 reference decision mismatch")
    if payload.get("uses_test_set") is not False or payload.get("held_out_test_authorized") is not False:
        raise ValueError("D1 reference is not train-only")
    if payload.get("diagnostic_tasks") != config["num_tasks"] or payload.get("class_order") != class_order:
        raise ValueError("D1 task stream mismatch")
    reference = config["reference_d1"]
    provenance = payload.get("provenance", {})
    if provenance.get("runner_git_dirty") is not False:
        raise ValueError("D1 reference was not produced from a clean worktree")
    for key in (
        "config_sha256", "runner_git_commit", "train_sha256",
        "training_indices_sha256", "validation_indices_sha256",
    ):
        if provenance.get(key) != reference[key]:
            raise ValueError(f"D1 provenance mismatch for {key}")
    if train_sha256 != reference["train_sha256"]:
        raise ValueError("runtime train artifact differs from D1")
    if _sequence_sha256(training_parts) != reference["training_indices_sha256"]:
        raise ValueError("runtime training split differs from D1")
    if _sequence_sha256(validation_parts) != reference["validation_indices_sha256"]:
        raise ValueError("runtime validation split differs from D1")
    matches = [item for item in payload.get("results", []) if item.get("method") == "srq_int8"]
    if len(matches) != 1:
        raise ValueError("D1 reference must contain exactly one SRQ result")
    srq = matches[0]
    if (
        srq.get("status") != "complete"
        or srq.get("uses_test_set") is not False
        or srq.get("exemplar_free") is not True
    ):
        raise ValueError("D1 SRQ result contract mismatch")
    checks = {
        "srq_persistent_state_bytes": srq.get("persistent_state_bytes"),
        "srq_validation_average_accuracy": srq.get("validation_average_accuracy"),
        "srq_final_accuracy": srq.get("stage_accuracy", [None])[-1],
    }
    for key, observed in checks.items():
        expected = reference[key]
        if observed is None or abs(float(observed) - float(expected)) > 1e-10:
            raise ValueError(f"D1 SRQ reference mismatch for {key}")
    return payload, srq


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    code_cache_dir = Path(args.code_cache_dir).resolve()
    d1_result_path = Path(args.d1_result).resolve()
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
    if int(train["features"].shape[1]) != config["feature_dim"]:
        raise ValueError("feature cache dimension differs from D2 config")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(range(config["num_classes"])):
        raise ValueError("training labels do not match locked classes")
    train_path = feature_cache_dir / "train.pt"
    train_sha256 = _sha256_file(train_path)
    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"], task_indices, config["seed"], config["validation_fraction"]
    )
    d1_payload, srq = _load_and_verify_d1(
        path=d1_result_path, config=config, train_sha256=train_sha256,
        class_order=class_order, training_parts=training_parts,
        validation_parts=validation_parts,
    )
    code_indices, code_values, code_metadata, projection = _prepare_code_cache(
        train=train, train_sha256=train_sha256, cache_dir=code_cache_dir,
        config=_cache_config(config), device=args.device,
    )
    git = _git_provenance()
    context = {
        "config_sha256": _sha256_file(config_path),
        "d1_result_sha256": _sha256_file(d1_result_path),
        "train_sha256": train_sha256,
        "code_identity": code_metadata["identity_sha256"],
        "projection_sha256": _tensor_content_sha256(projection),
        "training_indices_sha256": _sequence_sha256(training_parts),
        "validation_indices_sha256": _sequence_sha256(validation_parts),
        "runner_git_commit": git["git_commit"], "runner_git_dirty": git["git_dirty"],
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "d0_runner_source_sha256": _sha256_file(ROOT / "tools/srq_fly_d0.py"),
    }
    context_sha256 = _sha256_bytes(json.dumps(context, sort_keys=True).encode())
    output_dir.mkdir(parents=True, exist_ok=True)
    unit_path = _unit_path(output_dir, "exact_fly_4518")
    result = _load_unit(unit_path, context_sha256)
    if result is None:
        print("START exact_fly_4518", flush=True)
        result = _save_unit(
            unit_path, context_sha256,
            d0._evaluate_exact(
                name="exact_fly_4518", config=config,
                representation=config["representation"], train=train,
                code_indices=code_indices, code_values=code_values,
                projection=projection, training_parts=training_parts,
                validation_parts=validation_parts, device=torch.device(args.device),
            ),
        )
        print("DONE exact_fly_4518 status=complete", flush=True)
    state_mismatch = abs(
        result["persistent_state_bytes"] - srq["persistent_state_bytes"]
    ) / srq["persistent_state_bytes"]
    average_gain = srq["validation_average_accuracy"] - result["validation_average_accuracy"]
    final_gain = srq["stage_accuracy"][-1] - result["stage_accuracy"][-1]
    gates_config = config["gates"]
    gates = {
        "control_complete": result["status"] == "complete",
        "heldout_test_remained_hidden": not (feature_cache_dir / "test.pt").exists(),
        "numerical_stability": result["maximum_solver_relative_residual"] <= gates_config["maximum_solver_relative_residual"],
        "analytic_state_accounting_matches_runtime": result["persistent_state_bytes"] == config["expected_exact_state_bytes"],
        "state_budget_matched": state_mismatch <= gates_config["maximum_state_mismatch_fraction"],
        "srq_average_gain_within_gate": average_gain >= gates_config["minimum_srq_average_gain_pp"],
        "srq_final_gain_within_gate": final_gain >= gates_config["minimum_srq_final_gain_pp"],
    }
    decision = "PASS_REVIEW_D2" if all(gates.values()) else "STOP_SRQ_FLY_D2"
    payload = {
        "schema_version": 1, "study_id": config["study_id"], "status": decision,
        "uses_test_set": False, "held_out_test_authorized": False,
        "provenance": context, "class_order": class_order,
        "d1_reference": {
            "result_sha256": context["d1_result_sha256"],
            "d1_status": d1_payload.get("status"),
            "srq_validation_average_accuracy": srq["validation_average_accuracy"],
            "srq_final_accuracy": srq["stage_accuracy"][-1],
            "srq_persistent_state_bytes": srq["persistent_state_bytes"],
        },
        "state_matched_exact_fly": result,
        "comparison": {
            "state_mismatch_fraction": state_mismatch,
            "srq_average_gain_pp": average_gain,
            "srq_final_gain_pp": final_gain,
        },
        "gates": gates,
    }
    path = output_dir / "d2_results.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(f"SRQ-FLY D2 {decision}", flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--code-cache-dir", required=True)
    parser.add_argument("--d1-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-test-hidden", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
