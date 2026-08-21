"""Nested train-only lambda robustness control for state-matched SRQ-FLY D2."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import srq_fly_d0 as d0
from tools import srq_fly_d2_state_match as d2
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.tail_fly_phasea import _git_provenance, _load_unit, _save_unit, _unit_path
from tools.twa_fly_pilot import (
    _prepare_code_cache,
    _sequence_sha256,
    _sha256_bytes,
    _sha256_file,
    _tensor_content_sha256,
)


TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name", "feature_dim",
    "checkpoint_sha256", "seed", "num_classes", "num_tasks",
    "outer_validation_fraction", "inner_validation_fraction",
    "statistics_dtype", "selection_lambdas", "representation",
    "reference_d2", "expected_exact_state_bytes", "gates",
}
REFERENCE_KEYS = {
    "result_sha256", "config_sha256", "runner_git_commit", "train_sha256",
    "outer_training_indices_sha256", "outer_validation_indices_sha256",
    "srq_persistent_state_bytes", "srq_validation_average_accuracy",
    "srq_final_accuracy", "d2_exact_validation_average_accuracy",
    "d2_exact_final_accuracy", "d2_exact_persistent_state_bytes",
}
GATE_KEYS = {
    "maximum_solver_relative_residual", "maximum_state_mismatch_fraction",
    "minimum_srq_average_gain_pp", "minimum_srq_final_gain_pp",
}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config key: {key}")
        result[key] = value
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _read_config(path: Path) -> dict:
    config = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("config keys/schema mismatch")
    if config["seed"] != 2025:
        raise ValueError("new SRQ-FLY protocols require repository seed 2025")
    if (
        config["feature_dim"] <= 0
        or config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
    ):
        raise ValueError("invalid class/task dimensions")
    if not 0 < config["outer_validation_fraction"] < 1:
        raise ValueError("outer_validation_fraction must be in (0,1)")
    if not 0 < config["inner_validation_fraction"] < 1:
        raise ValueError("inner_validation_fraction must be in (0,1)")
    if config["statistics_dtype"] != "float32":
        raise ValueError("D2.1 state accounting requires float32 statistics")
    lambdas = config["selection_lambdas"]
    if (
        not isinstance(lambdas, list)
        or not lambdas
        or any(not isinstance(value, (int, float)) for value in lambdas)
        or any(not math.isfinite(float(value)) or float(value) <= 0 for value in lambdas)
        or list(map(float, lambdas)) != sorted(set(map(float, lambdas)))
        or 1_000_000.0 not in set(map(float, lambdas))
    ):
        raise ValueError("selection_lambdas must be sorted, unique, positive, and include 1e6")
    representation = config["representation"]
    if set(representation) != d0.REPRESENTATION_KEYS:
        raise ValueError("representation keys mismatch")
    if (
        min(
            representation["expand_dim"], representation["synaptic_degree"],
            representation["encode_batch_size"], representation["evaluation_batch_size"],
        ) <= 0
        or representation["synaptic_degree"] > config["feature_dim"]
        or not 0 < representation["coding_level"] <= 1
    ):
        raise ValueError("invalid representation")
    reference = config["reference_d2"]
    if set(reference) != REFERENCE_KEYS:
        raise ValueError("D2 reference keys mismatch")
    metrics = (
        reference["srq_persistent_state_bytes"],
        reference["srq_validation_average_accuracy"], reference["srq_final_accuracy"],
        reference["d2_exact_validation_average_accuracy"],
        reference["d2_exact_final_accuracy"], reference["d2_exact_persistent_state_bytes"],
    )
    if any(float(value) <= 0 for value in metrics):
        raise ValueError("invalid D2 reference metrics")
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
    analytic = d2.exact_fly_state_bytes(
        feature_dim=int(config["feature_dim"]),
        expand_dim=int(representation["expand_dim"]),
        synaptic_degree=int(representation["synaptic_degree"]),
        num_classes=int(config["num_classes"]),
    )
    if analytic != config["expected_exact_state_bytes"]:
        raise ValueError("expected exact state does not match analytic accounting")
    if analytic != reference["d2_exact_persistent_state_bytes"]:
        raise ValueError("D2 exact state differs from analytic accounting")
    lower = d2.exact_fly_state_bytes(
        feature_dim=int(config["feature_dim"]),
        expand_dim=int(representation["expand_dim"]) - 1,
        synaptic_degree=int(representation["synaptic_degree"]),
        num_classes=int(config["num_classes"]),
    )
    upper = d2.exact_fly_state_bytes(
        feature_dim=int(config["feature_dim"]),
        expand_dim=int(representation["expand_dim"]) + 1,
        synaptic_degree=int(representation["synaptic_degree"]),
        num_classes=int(config["num_classes"]),
    )
    target = reference["srq_persistent_state_bytes"]
    if analytic > target or target - analytic > min(abs(target - lower), abs(target - upper)):
        raise ValueError("representation is not the closest non-exceeding state match")
    return config


def _validate_nested_parts(
    outer_training: list[torch.Tensor], outer_validation: list[torch.Tensor],
    inner_fit: list[torch.Tensor], inner_validation: list[torch.Tensor],
) -> None:
    if not (
        len(outer_training) == len(outer_validation) == len(inner_fit) == len(inner_validation)
    ):
        raise ValueError("nested split task counts differ")
    for task, (outer_fit, outer_val, fit, val) in enumerate(zip(
        outer_training, outer_validation, inner_fit, inner_validation
    )):
        sets = [set(map(int, value.tolist())) for value in (outer_fit, outer_val, fit, val)]
        outer_fit_set, outer_val_set, fit_set, val_set = sets
        if not fit_set or not val_set:
            raise ValueError(f"empty inner partition at task {task}")
        if fit_set & val_set or outer_fit_set & outer_val_set:
            raise ValueError(f"overlapping nested partition at task {task}")
        if fit_set | val_set != outer_fit_set:
            raise ValueError(f"inner partitions do not cover outer training at task {task}")
        if (fit_set | val_set) & outer_val_set:
            raise ValueError(f"outer validation leaked into selection at task {task}")


def _verify_d2(
    *, path: Path, config: dict, train_sha256: str, class_order: list[int],
    outer_training: list[torch.Tensor], outer_validation: list[torch.Tensor],
) -> dict:
    reference = config["reference_d2"]
    if _sha256_file(path) != reference["result_sha256"]:
        raise ValueError("D2 result SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS_REVIEW_D2":
        raise ValueError("D2 did not authorize robustness review")
    if payload.get("uses_test_set") is not False or payload.get("held_out_test_authorized") is not False:
        raise ValueError("D2 reference is not train-only")
    if payload.get("class_order") != class_order:
        raise ValueError("D2 class order mismatch")
    provenance = payload.get("provenance", {})
    expected_provenance = {
        "config_sha256": reference["config_sha256"],
        "runner_git_commit": reference["runner_git_commit"],
        "train_sha256": reference["train_sha256"],
        "training_indices_sha256": reference["outer_training_indices_sha256"],
        "validation_indices_sha256": reference["outer_validation_indices_sha256"],
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(f"D2 provenance mismatch for {key}")
    if provenance.get("runner_git_dirty") is not False:
        raise ValueError("D2 reference was produced from a dirty worktree")
    if train_sha256 != reference["train_sha256"]:
        raise ValueError("runtime train artifact differs from D2")
    if _sequence_sha256(outer_training) != reference["outer_training_indices_sha256"]:
        raise ValueError("runtime outer training split differs from D2")
    if _sequence_sha256(outer_validation) != reference["outer_validation_indices_sha256"]:
        raise ValueError("runtime outer validation split differs from D2")
    if not payload.get("gates") or not all(payload["gates"].values()):
        raise ValueError("D2 reference gates were not all true")
    exact = payload.get("state_matched_exact_fly", {})
    srq = payload.get("d1_reference", {})
    checks = {
        "d2_exact_validation_average_accuracy": exact.get("validation_average_accuracy"),
        "d2_exact_final_accuracy": exact.get("stage_accuracy", [None])[-1],
        "d2_exact_persistent_state_bytes": exact.get("persistent_state_bytes"),
        "srq_validation_average_accuracy": srq.get("srq_validation_average_accuracy"),
        "srq_final_accuracy": srq.get("srq_final_accuracy"),
        "srq_persistent_state_bytes": srq.get("srq_persistent_state_bytes"),
    }
    for key, observed in checks.items():
        if observed is None or abs(float(observed) - float(reference[key])) > 1e-10:
            raise ValueError(f"D2 metric mismatch for {key}")
    if (
        exact.get("status") != "complete"
        or exact.get("uses_test_set") is not False
        or exact.get("exemplar_free") is not True
    ):
        raise ValueError("D2 exact control contract mismatch")
    return payload


def _candidate_name(index: int, ridge_lambda: float) -> str:
    value = f"{ridge_lambda:.0e}".replace("+", "")
    return f"inner_{index:02d}_lambda_{value}"


def _evaluate(
    *, name: str, ridge_lambda: float, config: dict, train: dict,
    code_indices: torch.Tensor, code_values: torch.Tensor, projection: torch.Tensor,
    training_parts: list[torch.Tensor], validation_parts: list[torch.Tensor],
    device: torch.device,
) -> dict:
    candidate_config = {**config, "ridge_lambda": float(ridge_lambda)}
    return d0._evaluate_exact(
        name=name, config=candidate_config, representation=config["representation"],
        train=train, code_indices=code_indices, code_values=code_values,
        projection=projection, training_parts=training_parts,
        validation_parts=validation_parts, device=device,
    )


def _choose_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("no lambda candidates")
    return min(
        candidates,
        key=lambda item: (-float(item["validation_average_accuracy"]), float(item["ridge_lambda"])),
    )


def _validate_exact_result(
    result: dict, *, name: str, ridge_lambda: float, num_tasks: int,
    expected_state_bytes: int,
) -> None:
    stages = result.get("stage_accuracy")
    forbidden = ("sample", "feature", "label", "code", "history")
    if (
        any(any(token in str(key).lower() for token in forbidden) for key in result)
        or result.get("method") != name
        or result.get("status") != "complete"
        or result.get("uses_test_set") is not False
        or result.get("exemplar_free") is not True
        or float(result.get("ridge_lambda", -1)) != float(ridge_lambda)
        or result.get("persistent_state_bytes") != expected_state_bytes
        or not isinstance(stages, list)
        or len(stages) != num_tasks
        or any(not math.isfinite(float(value)) for value in stages)
    ):
        raise ValueError(f"invalid exact-FLY result contract for {name}")
    average = float(result.get("validation_average_accuracy", float("nan")))
    residual = float(result.get("maximum_solver_relative_residual", float("nan")))
    if (
        not math.isfinite(average)
        or abs(average - sum(map(float, stages)) / len(stages)) > 1e-10
        or not math.isfinite(residual)
        or residual < 0
    ):
        raise ValueError(f"invalid exact-FLY metrics for {name}")


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    code_cache_dir = Path(args.code_cache_dir).resolve()
    d2_result_path = Path(args.d2_result).resolve()
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
        raise ValueError("feature cache dimension differs from D2.1 config")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(range(config["num_classes"])):
        raise ValueError("training labels do not match locked classes")
    train_sha256 = _sha256_file(feature_cache_dir / "train.pt")
    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    outer_training, outer_validation = train_validation_indices(
        train["labels"], task_indices, config["seed"], config["outer_validation_fraction"]
    )
    d2_payload = _verify_d2(
        path=d2_result_path, config=config, train_sha256=train_sha256,
        class_order=class_order, outer_training=outer_training,
        outer_validation=outer_validation,
    )
    inner_fit, inner_validation = train_validation_indices(
        train["labels"], outer_training, config["seed"], config["inner_validation_fraction"]
    )
    _validate_nested_parts(outer_training, outer_validation, inner_fit, inner_validation)
    code_indices, code_values, code_metadata, projection = _prepare_code_cache(
        train=train, train_sha256=train_sha256, cache_dir=code_cache_dir,
        config=d2._cache_config({**config, "ridge_lambda": 1_000_000.0}),
        device=args.device,
    )
    git = _git_provenance()
    context = {
        "config_sha256": _sha256_file(config_path),
        "d2_result_sha256": _sha256_file(d2_result_path),
        "train_sha256": train_sha256,
        "code_identity": code_metadata["identity_sha256"],
        "projection_sha256": _tensor_content_sha256(projection),
        "outer_training_indices_sha256": _sequence_sha256(outer_training),
        "outer_validation_indices_sha256": _sequence_sha256(outer_validation),
        "inner_fit_indices_sha256": _sequence_sha256(inner_fit),
        "inner_validation_indices_sha256": _sequence_sha256(inner_validation),
        "runner_git_commit": git["git_commit"], "runner_git_dirty": git["git_dirty"],
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "d0_runner_source_sha256": _sha256_file(ROOT / "tools/srq_fly_d0.py"),
    }
    context_sha256 = _sha256_bytes(json.dumps(context, sort_keys=True).encode())
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    candidates = []
    for index, ridge_lambda in enumerate(map(float, config["selection_lambdas"])):
        name = _candidate_name(index, ridge_lambda)
        unit_path = _unit_path(output_dir, name)
        result = _load_unit(unit_path, context_sha256)
        if result is None:
            print(f"INNER START lambda={ridge_lambda:g}", flush=True)
            result = _save_unit(
                unit_path, context_sha256,
                _evaluate(
                    name=name, ridge_lambda=ridge_lambda, config=config, train=train,
                    code_indices=code_indices, code_values=code_values,
                    projection=projection, training_parts=inner_fit,
                    validation_parts=inner_validation, device=device,
                ),
            )
            print(
                f"INNER DONE lambda={ridge_lambda:g} "
                f"AA={result['validation_average_accuracy']:.6f}", flush=True,
            )
        _validate_exact_result(
            result, name=name, ridge_lambda=ridge_lambda,
            num_tasks=config["num_tasks"],
            expected_state_bytes=config["expected_exact_state_bytes"],
        )
        candidates.append(result)
    selected = _choose_candidate(candidates)
    selection_payload = {
        "schema_version": 1,
        "selection_protocol": "nested inner split of D1 outer-training only",
        "uses_outer_validation_for_selection": False,
        "uses_test_set": False,
        "context_sha256": context_sha256,
        "inner_fit_indices_sha256": context["inner_fit_indices_sha256"],
        "inner_validation_indices_sha256": context["inner_validation_indices_sha256"],
        "tie_break": "maximum inner AA, then smaller lambda",
        "selected_lambda": selected["ridge_lambda"],
        "candidates": candidates,
    }
    selection_path = output_dir / "lambda_selection.json"
    _atomic_json(selection_path, selection_payload)
    print(f"LOCKED lambda={selected['ridge_lambda']:g}", flush=True)
    outer_context = {
        **context,
        "lambda_selection_sha256": _sha256_file(selection_path),
        "selected_lambda": selected["ridge_lambda"],
    }
    outer_context_sha256 = _sha256_bytes(json.dumps(outer_context, sort_keys=True).encode())
    outer_path = _unit_path(output_dir, "outer_exact_fly_4518")
    outer = _load_unit(outer_path, outer_context_sha256)
    if outer is None:
        print(f"OUTER START locked_lambda={selected['ridge_lambda']:g}", flush=True)
        outer = _save_unit(
            outer_path, outer_context_sha256,
            _evaluate(
                name="outer_exact_fly_4518", ridge_lambda=selected["ridge_lambda"],
                config=config, train=train, code_indices=code_indices,
                code_values=code_values, projection=projection,
                training_parts=outer_training, validation_parts=outer_validation,
                device=device,
            ),
        )
        print(
            f"OUTER DONE AA={outer['validation_average_accuracy']:.6f}", flush=True
        )
    _validate_exact_result(
        outer, name="outer_exact_fly_4518",
        ridge_lambda=selected["ridge_lambda"], num_tasks=config["num_tasks"],
        expected_state_bytes=config["expected_exact_state_bytes"],
    )
    reference = config["reference_d2"]
    state_mismatch = abs(
        outer["persistent_state_bytes"] - reference["srq_persistent_state_bytes"]
    ) / reference["srq_persistent_state_bytes"]
    average_gain = (
        reference["srq_validation_average_accuracy"] - outer["validation_average_accuracy"]
    )
    final_gain = reference["srq_final_accuracy"] - outer["stage_accuracy"][-1]
    gates_config = config["gates"]
    maximum_inner_residual = max(
        float(item["maximum_solver_relative_residual"]) for item in candidates
    )
    gates = {
        "selection_complete": len(candidates) == len(config["selection_lambdas"]),
        "selected_lambda_is_locked_candidate": selected["ridge_lambda"] in list(
            map(float, config["selection_lambdas"])
        ),
        "outer_validation_not_used_for_selection": selection_payload[
            "uses_outer_validation_for_selection"
        ] is False,
        "heldout_test_remained_hidden": not (feature_cache_dir / "test.pt").exists(),
        "inner_numerical_stability": maximum_inner_residual
        <= gates_config["maximum_solver_relative_residual"],
        "outer_numerical_stability": outer["maximum_solver_relative_residual"]
        <= gates_config["maximum_solver_relative_residual"],
        "analytic_state_accounting_matches_runtime": outer["persistent_state_bytes"]
        == config["expected_exact_state_bytes"],
        "state_budget_matched": state_mismatch
        <= gates_config["maximum_state_mismatch_fraction"],
        "srq_average_gain_within_gate": average_gain
        >= gates_config["minimum_srq_average_gain_pp"],
        "srq_final_gain_within_gate": final_gain
        >= gates_config["minimum_srq_final_gain_pp"],
    }
    decision = "PASS_REVIEW_D21" if all(gates.values()) else "STOP_SRQ_FLY_D21"
    payload = {
        "schema_version": 1, "study_id": config["study_id"], "status": decision,
        "uses_test_set": False, "held_out_test_authorized": False,
        "uses_outer_validation_for_selection": False,
        "provenance": outer_context, "class_order": class_order,
        "d2_reference": {
            "result_sha256": reference["result_sha256"],
            "d2_status": d2_payload["status"],
            "srq_validation_average_accuracy": reference[
                "srq_validation_average_accuracy"
            ],
            "srq_final_accuracy": reference["srq_final_accuracy"],
            "srq_persistent_state_bytes": reference["srq_persistent_state_bytes"],
        },
        "selection": {
            "selected_lambda": selected["ridge_lambda"],
            "selected_inner_average_accuracy": selected["validation_average_accuracy"],
            "candidate_count": len(candidates),
            "maximum_inner_solver_relative_residual": maximum_inner_residual,
            "selection_artifact_sha256": outer_context["lambda_selection_sha256"],
        },
        "tuned_state_matched_exact_fly": outer,
        "comparison": {
            "state_mismatch_fraction": state_mismatch,
            "srq_average_gain_pp": average_gain,
            "srq_final_gain_pp": final_gain,
        },
        "gates": gates,
    }
    _atomic_json(output_dir / "d21_results.json", payload)
    print(f"SRQ-FLY D2.1 {decision}", flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--code-cache-dir", required=True)
    parser.add_argument("--d2-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-test-hidden", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
