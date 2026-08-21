"""Locked CIFAR-100 train-only selection gate before held-out evaluation."""

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

from methods.srq_fly.storage import projected_srq_state_bytes
from tools import srq_fly_d1 as d1
from tools import srq_fly_d3_cub as d3
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.srq_fly_d2_state_match import exact_fly_state_bytes
from tools.srq_fly_d21_lambda_robustness import _validate_nested_parts
from tools.tail_fly_phasea import _git_provenance, _load_unit, _save_unit, _unit_path
from tools.twa_fly_pilot import (
    _prepare_code_cache,
    _sequence_sha256,
    _sha256_bytes,
    _sha256_file,
    _tensor_content_sha256,
)


TOP_KEYS = {
    "schema_version", "study_id", "dataset", "dataset_version", "model_name",
    "feature_dim", "checkpoint_sha256", "seed", "num_classes", "num_tasks",
    "train_samples", "test_samples", "outer_validation_fraction",
    "inner_validation_fraction", "statistics_dtype", "solver_dtype",
    "raw_statistics_dtype", "selection_lambdas", "fixed_raw_ridge_lambda",
    "large_representation", "matched_representation", "storage",
    "expected_state", "gates",
}
REPRESENTATION_KEYS = {
    "expand_dim", "synaptic_degree", "coding_level", "encode_batch_size",
    "evaluation_batch_size",
}
EXPECTED_STATE_KEYS = {
    "nominal_large_projection_entries", "nominal_matched_projection_entries",
    "maximum_missing_projection_entries", "nominal_exact_large_bytes",
    "nominal_srq_large_bytes", "nominal_exact_matched_bytes",
}
GATE_KEYS = {
    "maximum_inner_solver_relative_residual",
    "maximum_outer_solver_relative_residual",
    "maximum_average_gap_to_exact_large_pp",
    "maximum_final_gap_to_exact_large_pp", "minimum_prediction_agreement",
    "maximum_state_fraction_of_exact_large", "maximum_state_mismatch_fraction",
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
    if config["dataset"] != "CIFAR-100" or config["seed"] != 2025:
        raise ValueError("D5 is locked to CIFAR-100 and repository seed 2025")
    if (
        config["feature_dim"] <= 0 or config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
        or min(config["train_samples"], config["test_samples"]) <= 0
    ):
        raise ValueError("invalid dataset dimensions")
    if not 0 < config["outer_validation_fraction"] < 1 or not 0 < config[
        "inner_validation_fraction"
    ] < 1:
        raise ValueError("invalid nested validation fractions")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("SRQ state contract requires float32 statistics/solver")
    if config["raw_statistics_dtype"] != "float64":
        raise ValueError("raw Ridge control requires float64 statistics")
    lambdas = list(map(float, config["selection_lambdas"]))
    if (
        not lambdas or lambdas != sorted(set(lambdas))
        or any(not math.isfinite(value) or value <= 0 for value in lambdas)
        or config["fixed_raw_ridge_lambda"] <= 0
    ):
        raise ValueError("invalid Ridge parameters")
    for name in ("large_representation", "matched_representation"):
        representation = config[name]
        if set(representation) != REPRESENTATION_KEYS:
            raise ValueError(f"{name} keys mismatch")
        if (
            min(
                representation["expand_dim"], representation["synaptic_degree"],
                representation["encode_batch_size"],
                representation["evaluation_batch_size"],
            ) <= 0
            or representation["synaptic_degree"] > config["feature_dim"]
            or not 0 < representation["coding_level"] <= 1
        ):
            raise ValueError(f"invalid {name}")
    if set(config["storage"]) != {"block_size", "group_size"} or min(
        config["storage"].values()
    ) <= 0:
        raise ValueError("invalid storage")
    if set(config["expected_state"]) != EXPECTED_STATE_KEYS:
        raise ValueError("expected_state keys mismatch")
    if set(config["gates"]) != GATE_KEYS or any(
        float(value) <= 0 for value in config["gates"].values()
    ):
        raise ValueError("invalid gates")

    large = config["large_representation"]
    matched = config["matched_representation"]
    storage = config["storage"]
    expected = config["expected_state"]
    projected = projected_srq_state_bytes(
        feature_dim=config["feature_dim"], expand_dim=large["expand_dim"],
        synaptic_degree=large["synaptic_degree"], num_classes=config["num_classes"],
        block_size=storage["block_size"], group_size=storage["group_size"],
    )
    analytic = {
        "nominal_large_projection_entries": large["expand_dim"] * large["synaptic_degree"],
        "nominal_matched_projection_entries": matched["expand_dim"] * matched["synaptic_degree"],
        "nominal_exact_large_bytes": projected["exact_fly_total_bytes"],
        "nominal_srq_large_bytes": projected["compressed_total_bytes"],
        "nominal_exact_matched_bytes": exact_fly_state_bytes(
            feature_dim=config["feature_dim"], expand_dim=matched["expand_dim"],
            synaptic_degree=matched["synaptic_degree"], num_classes=config["num_classes"],
        ),
    }
    for key, value in analytic.items():
        if expected[key] != value:
            raise ValueError(f"expected state is inconsistent for {key}")
    target = expected["nominal_srq_large_bytes"]
    current = expected["nominal_exact_matched_bytes"]
    next_state = exact_fly_state_bytes(
        feature_dim=config["feature_dim"], expand_dim=matched["expand_dim"] + 1,
        synaptic_degree=matched["synaptic_degree"], num_classes=config["num_classes"],
    )
    if current > target or next_state <= target:
        raise ValueError("matched dimension is not closest non-exceeding SRQ state")
    return config


def _cache_config(config: dict, representation: dict) -> dict:
    return {
        "seed": config["seed"], "num_classes": config["num_classes"],
        "representation": dict(representation), "statistics_dtype": "float32",
        "raw_ridge_lambda": 1.0, "solver_tolerance": 1e-5,
        "solver_max_iterations": 100,
    }


def _validate_train_cache(config: dict, cache_dir: Path):
    train, _, metadata = validate_cache(
        cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    expected_metadata = {
        "dataset_version": config["dataset_version"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "feature_dim": config["feature_dim"],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"feature-cache metadata mismatch for {key}")
    if tuple(train["features"].shape) != (config["train_samples"], config["feature_dim"]):
        raise ValueError("feature-cache train shape mismatch")
    if metadata.get("split_sizes", {}).get("test") != config["test_samples"]:
        raise ValueError("feature-cache declared test size mismatch")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(
        range(config["num_classes"])
    ):
        raise ValueError("training labels do not cover locked CIFAR classes")
    return train, metadata


def _runtime_state(config: dict, large_projection: torch.Tensor, matched_projection: torch.Tensor) -> dict:
    expected = config["expected_state"]
    large_nnz = int(large_projection.values().numel())
    matched_nnz = int(matched_projection.values().numel())
    missing_large = expected["nominal_large_projection_entries"] - large_nnz
    missing_matched = expected["nominal_matched_projection_entries"] - matched_nnz
    if (
        min(missing_large, missing_matched) < 0
        or max(missing_large, missing_matched) > expected["maximum_missing_projection_entries"]
    ):
        raise ValueError("projection stored-entry count is outside locked bounds")
    return {
        "large_projection_nonzeros": large_nnz,
        "matched_projection_nonzeros": matched_nnz,
        "missing_large_projection_entries": missing_large,
        "missing_matched_projection_entries": missing_matched,
        "exact_large_bytes": expected["nominal_exact_large_bytes"] - 12 * missing_large,
        "srq_large_bytes": expected["nominal_srq_large_bytes"] - 12 * missing_large,
        "exact_matched_bytes": expected["nominal_exact_matched_bytes"] - 12 * missing_matched,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    if (cache_dir / "test.pt").exists():
        raise RuntimeError("D5 train-only gate refuses a visible test.pt")
    train, metadata = _validate_train_cache(config, cache_dir)
    train_sha = _sha256_file(cache_dir / "train.pt")
    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    tasks = split(train["labels"], class_order, config["num_tasks"])
    outer_fit, outer_validation = train_validation_indices(
        train["labels"], tasks, config["seed"], config["outer_validation_fraction"]
    )
    inner_fit, inner_validation = train_validation_indices(
        train["labels"], outer_fit, config["seed"], config["inner_validation_fraction"]
    )
    _validate_nested_parts(outer_fit, outer_validation, inner_fit, inner_validation)
    print("CACHE START FLY-10000", flush=True)
    large = _prepare_code_cache(
        train=train, train_sha256=train_sha,
        cache_dir=Path(args.large_code_cache_dir).resolve(),
        config=_cache_config(config, config["large_representation"]),
        device=args.device,
    )
    print("CACHE START FLY-4409", flush=True)
    matched = _prepare_code_cache(
        train=train, train_sha256=train_sha,
        cache_dir=Path(args.matched_code_cache_dir).resolve(),
        config=_cache_config(config, config["matched_representation"]),
        device=args.device,
    )
    prefix = d3._verify_projection_prefix(large[3], matched[3])
    runtime_state = _runtime_state(config, large[3], matched[3])
    git = _git_provenance()
    context = {
        "config_sha256": _sha256_file(config_path), "train_sha256": train_sha,
        "large_code_identity": large[2]["identity_sha256"],
        "matched_code_identity": matched[2]["identity_sha256"],
        "large_projection_sha256": _tensor_content_sha256(large[3]),
        "matched_projection_sha256": _tensor_content_sha256(matched[3]),
        "outer_fit_indices_sha256": _sequence_sha256(outer_fit),
        "outer_validation_indices_sha256": _sequence_sha256(outer_validation),
        "inner_fit_indices_sha256": _sequence_sha256(inner_fit),
        "inner_validation_indices_sha256": _sequence_sha256(inner_validation),
        "runner_git_commit": git["git_commit"], "runner_git_dirty": git["git_dirty"],
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "learner_source_sha256": _sha256_file(ROOT / "methods/srq_fly/learner.py"),
        "storage_source_sha256": _sha256_file(ROOT / "methods/srq_fly/storage.py"),
    }
    context_sha = _sha256_bytes(json.dumps(context, sort_keys=True).encode())
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    candidates = []
    for index, ridge_lambda in enumerate(map(float, config["selection_lambdas"])):
        name = d3._candidate_name("cifar_fly10000", index, ridge_lambda)
        path = _unit_path(output_dir, name)
        result = _load_unit(path, context_sha)
        if result is None:
            print(f"INNER START lambda={ridge_lambda:g}", flush=True)
            result = _save_unit(path, context_sha, d3._evaluate_exact(
                name=name, ridge_lambda=ridge_lambda, config=config,
                representation=config["large_representation"], train=train,
                code_indices=large[0], code_values=large[1], projection=large[3],
                training_parts=inner_fit, validation_parts=inner_validation,
                device=device,
            ))
            print(
                f"INNER DONE lambda={ridge_lambda:g} "
                f"AA={result['validation_average_accuracy']:.6f}", flush=True,
            )
        d3._validate_result(
            result, name=name, ridge_lambda=ridge_lambda,
            num_tasks=config["num_tasks"],
            expected_state_bytes=runtime_state["exact_large_bytes"],
        )
        candidates.append(result)
    selected = d3._choose_candidate(candidates)
    selection = {
        "schema_version": 1,
        "protocol": "nested inner split of CIFAR-100 training data only",
        "uses_outer_validation_for_selection": False, "uses_test_set": False,
        "tie_break": "maximum inner AA, then smaller lambda",
        "selected_fly_and_srq_lambda": selected["ridge_lambda"],
        "fixed_raw_ridge_lambda": config["fixed_raw_ridge_lambda"],
        "candidates": candidates,
    }
    selection_path = output_dir / "lambda_selection.json"
    _atomic_json(selection_path, selection)
    print(f"LOCKED fly/srq lambda={selected['ridge_lambda']:g}", flush=True)
    outer_context = {
        **context, "selection_sha256": _sha256_file(selection_path),
        "selected_fly_and_srq_lambda": selected["ridge_lambda"],
    }
    outer_context_sha = _sha256_bytes(json.dumps(outer_context, sort_keys=True).encode())
    paired_path = _unit_path(output_dir, "outer_paired_exact_srq_10000")
    paired = _load_unit(paired_path, outer_context_sha)
    if paired is None:
        print("OUTER START paired exact/SRQ FLY-10000", flush=True)
        paired = _save_unit(
            paired_path, outer_context_sha,
            d1._evaluate_paired_exact_srq(
                config=d3._paired_config(config, selected["ridge_lambda"]),
                train=train, code_indices=large[0], code_values=large[1],
                projection=large[3], training_parts=outer_fit,
                validation_parts=outer_validation, device=device,
            ),
        )
        print("OUTER DONE paired exact/SRQ FLY-10000", flush=True)
    exact, srq = paired["exact"], paired["srq"]
    d3._validate_result(
        exact, name="exact_fly_10000", ridge_lambda=selected["ridge_lambda"],
        num_tasks=config["num_tasks"], expected_state_bytes=runtime_state["exact_large_bytes"],
    )
    d3._validate_result(
        srq, name="srq_int8", ridge_lambda=selected["ridge_lambda"],
        num_tasks=config["num_tasks"], expected_state_bytes=runtime_state["srq_large_bytes"],
    )

    def outer_unit(name, evaluator, ridge_lambda, expected_state=None):
        path = _unit_path(output_dir, name)
        result = _load_unit(path, outer_context_sha)
        if result is None:
            print(f"OUTER START {name}", flush=True)
            result = _save_unit(path, outer_context_sha, evaluator())
            print(f"OUTER DONE {name}", flush=True)
        d3._validate_result(
            result, name=name, ridge_lambda=ridge_lambda,
            num_tasks=config["num_tasks"], expected_state_bytes=expected_state,
        )
        return result

    matched_result = outer_unit(
        "exact_fly_4409",
        lambda: d3._evaluate_exact(
            name="exact_fly_4409", ridge_lambda=selected["ridge_lambda"],
            config=config, representation=config["matched_representation"],
            train=train, code_indices=matched[0], code_values=matched[1],
            projection=matched[3], training_parts=outer_fit,
            validation_parts=outer_validation, device=device,
        ),
        selected["ridge_lambda"], runtime_state["exact_matched_bytes"],
    )
    raw = outer_unit(
        "raw_ridge",
        lambda: d3._evaluate_raw(
            name="raw_ridge", ridge_lambda=config["fixed_raw_ridge_lambda"],
            config=config, train=train, training_parts=outer_fit,
            validation_parts=outer_validation, device=device,
        ),
        config["fixed_raw_ridge_lambda"],
    )
    residuals = [
        *(float(item["maximum_solver_relative_residual"]) for item in candidates),
        float(exact["maximum_solver_relative_residual"]),
        float(srq["maximum_solver_relative_residual"]),
        float(matched_result["maximum_solver_relative_residual"]),
        float(raw["maximum_solver_relative_residual"]),
    ]
    gates_config = config["gates"]
    average_difference = srq["validation_average_accuracy"] - exact["validation_average_accuracy"]
    final_difference = srq["stage_accuracy"][-1] - exact["stage_accuracy"][-1]
    state_mismatch = abs(
        srq["persistent_state_bytes"] - matched_result["persistent_state_bytes"]
    ) / srq["persistent_state_bytes"]
    gates = {
        "selection_complete": len(candidates) == len(config["selection_lambdas"]),
        "outer_validation_not_used_for_selection": True,
        "heldout_test_remained_hidden": not (cache_dir / "test.pt").exists(),
        "projection_prefix_verified": prefix["verified"],
        "inner_numerical_stability": max(
            float(item["maximum_solver_relative_residual"]) for item in candidates
        ) <= gates_config["maximum_inner_solver_relative_residual"],
        "outer_numerical_stability": max(residuals[len(candidates):])
        <= gates_config["maximum_outer_solver_relative_residual"],
        "srq_tracks_exact_large_average": -average_difference
        <= gates_config["maximum_average_gap_to_exact_large_pp"],
        "srq_tracks_exact_large_final": -final_difference
        <= gates_config["maximum_final_gap_to_exact_large_pp"],
        "prediction_agreement": paired["minimum_prediction_agreement"]
        >= gates_config["minimum_prediction_agreement"],
        "compressed_state_fraction": srq["persistent_state_bytes"]
        / exact["persistent_state_bytes"]
        <= gates_config["maximum_state_fraction_of_exact_large"],
        "state_budget_matched": state_mismatch
        <= gates_config["maximum_state_mismatch_fraction"],
    }
    decision = "PASS_REVIEW_CIFAR_D5" if all(gates.values()) else "STOP_SRQ_FLY_CIFAR_D5"
    payload = {
        "schema_version": 1, "study_id": config["study_id"], "status": decision,
        "uses_test_set": False, "held_out_test_authorized": False,
        "provenance": outer_context, "source_feature_metadata": metadata,
        "class_order": class_order, "runtime_state": runtime_state,
        "selected_fly_and_srq_lambda": selected["ridge_lambda"],
        "fixed_raw_ridge_lambda": config["fixed_raw_ridge_lambda"],
        "exact_fly_10000": exact, "srq_fly_10000": srq,
        "exact_fly_4409": matched_result, "raw_ridge": raw,
        "comparison": {
            "srq_average_difference_from_exact_large_pp": average_difference,
            "srq_final_difference_from_exact_large_pp": final_difference,
            "srq_average_gain_over_state_matched_fly_pp":
                srq["validation_average_accuracy"] - matched_result["validation_average_accuracy"],
            "srq_final_gain_over_state_matched_fly_pp":
                srq["stage_accuracy"][-1] - matched_result["stage_accuracy"][-1],
            "srq_average_gain_over_raw_ridge_pp":
                srq["validation_average_accuracy"] - raw["validation_average_accuracy"],
            "state_mismatch_fraction": state_mismatch,
            "minimum_prediction_agreement": paired["minimum_prediction_agreement"],
        },
        "gates": gates,
    }
    _atomic_json(output_dir / "d5_results.json", payload)
    print(f"SRQ-FLY CIFAR D5 {decision}", flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs/srq_fly_cifar100_d5_train_only.json")
    )
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--large-code-cache-dir", required=True)
    parser.add_argument("--matched-code-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
